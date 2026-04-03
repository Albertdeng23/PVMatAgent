import os
import sys
import time
import json

# 假设你的项目根目录在 sys.path 中，这里导入我们规划好的各个专家模块
# 注意：这里为了兼容你之前的代码，我使用了你现有的类名
try:
    from tools_MPsearch import tools_MPsearch
    from tools_CHGnet import tools_CHGnet
    from src.physics.pv_calculators import SQCalculator
except ImportError:
    print("⚠️ 导入专家模块失败，请确保文件路径正确且环境变量已设置。")

class HighThroughputScreening:
    """
    自动化高通量筛选流水线。
    将 MP数据库检索 -> CHGNet结构弛豫与稳定性分析 -> S-Q光电效率极限计算 串联为闭环。
    """

    def __init__(self, cif_cache_dir="./cache/cifs"):
        print("⚙️ [Workflow] 初始化高通量筛选流水线...")
        self.cif_cache_dir = cif_cache_dir
        os.makedirs(self.cif_cache_dir, exist_ok=True)
        
        # 懒加载机制：按需初始化厚重的模型，避免启动过慢
        self.mp_engine = None
        self.chgnet_engine = None
        self.sq_engine = None

    def _init_engines(self):
        """确保所有底层物理与数据库引擎已加载"""
        if not self.mp_engine:
            self.mp_engine = tools_MPsearch()
        if not self.chgnet_engine:
            self.chgnet_engine = tools_CHGnet()
        if not self.sq_engine:
            self.sq_engine = SQCalculator()

    def run_evaluation(self, formula: str) -> dict:
        """
        核心闭环逻辑：一键评估材料的光伏潜力。
        :param formula: 化学式 (如 "CH3NH3PbI3")
        :return: 评估报告字典
        """
        self._init_engines()
        print(f"\n🚀 [Workflow] 开始全自动评估流程: {formula}")
        start_time = time.time()
        report = {
            "Formula": formula,
            "Status": "Failed",
            "Steps_Completed": [],
            "Data": {}
        }

        # =======================================================
        # Step 1: 外部数据库检索 (结构与基础电子性质)
        # =======================================================
        print(f"  -> [Step 1] 检索 Materials Project 获取 {formula} 的最稳定相...")
        mp_results = self.mp_engine.search_material(formula, limit=1, save_dir=self.cif_cache_dir)
        
        if not mp_results:
            report["Error"] = f"在 MP 数据库中未找到 {formula} 的稳定结构。"
            return report
            
        best_candidate = mp_results[0]
        cif_path = best_candidate.get("cif_path")
        mp_bandgap = best_candidate.get("band_gap_mp")
        crystal_sys = best_candidate.get("crystal_system")
        
        report["Steps_Completed"].append("MP_Search")
        report["Data"]["Crystal_System"] = crystal_sys
        report["Data"]["Bandgap_MP_eV"] = mp_bandgap
        report["Data"]["CIF_Path"] = cif_path

        # 检查是否具备光伏潜力的基础带隙 (通常光伏带隙在 0.5 - 3.0 eV 之间)
        if mp_bandgap is None or mp_bandgap < 0.1 or mp_bandgap > 4.0:
            report["Status"] = "Filtered_Out"
            report["Reason"] = f"带隙 ({mp_bandgap} eV) 不在典型光伏材料吸收范围内。"
            return report

        # =======================================================
        # Step 2: CHGNet 物理引擎分析 (热力学稳定性 & 几何弛豫)
        # =======================================================
        print(f"  -> [Step 2] 调用 CHGNet 进行毫秒级结构评估与形成能计算...")
        # 请求形成能计算任务
        chgnet_res = self.chgnet_engine.predict_properties(cif_path, task_list=['formation', 'relax'])
        
        if "error" in chgnet_res:
            report["Error"] = f"CHGNet 计算失败: {chgnet_res['error']}"
            return report
            
        e_form = chgnet_res.get("formation_energy_eV_per_atom", 0)
        is_stable = chgnet_res.get("is_thermodynamically_stable", False)
        
        report["Steps_Completed"].append("CHGNet_Analysis")
        report["Data"]["Formation_Energy_eV_atom"] = round(e_form, 4) if e_form else "N/A"
        report["Data"]["Is_Stable"] = is_stable

        if not is_stable:
            report["Status"] = "Filtered_Out"
            report["Reason"] = f"热力学不稳定 (形成能 > 0)，极易分解。"
            # 为了研究完整性，我们继续算效率，但标记为不稳定

        # =======================================================
        # Step 3: S-Q 理论极限计算 (光电性能预测)
        # =======================================================
        print(f"  -> [Step 3] 基于带隙 ({mp_bandgap} eV) 计算 S-Q 光电转化效率极限...")
        sq_res = self.sq_engine.evaluate_performance(mp_bandgap)
        
        if "error" in sq_res:
            report["Error"] = f"S-Q 计算失败: {sq_res['error']}"
            return report
            
        report["Steps_Completed"].append("SQ_Limit_Calculation")
        report["Data"]["Theoretical_PCE_%"] = sq_res.get("PCE_percent")
        report["Data"]["Voc_V"] = sq_res.get("Voc_V")
        report["Data"]["Jsc_mA_cm2"] = sq_res.get("Jsc_mA_cm2")
        report["Data"]["FF_%"] = sq_res.get("FF_percent")

        # =======================================================
        # 终审评估
        # =======================================================
        report["Status"] = "Success"
        cost_time = round(time.time() - start_time, 2)
        print(f"✅ [Workflow] 评估完成! 耗时: {cost_time} 秒")
        
        # 为 LLM 生成一段自然语言摘要
        summary = (
            f"材料 {formula} ({crystal_sys}相) 评估完毕。\n"
            f"- 热力学稳定性: {'稳定✅' if is_stable else '不稳定❌'} (形成能: {report['Data']['Formation_Energy_eV_atom']} eV/atom)。\n"
            f"- 电子结构: 带隙为 {mp_bandgap} eV。\n"
            f"- 光伏极限性能: 理论最高光电转换效率 (PCE) 可达 {report['Data']['Theoretical_PCE_%']}%, "
            f"开路电压 (Voc) 为 {report['Data']['Voc_V']} V, 短路电流 (Jsc) 为 {report['Data']['Jsc_mA_cm2']} mA/cm²。"
        )
        report["LLM_Summary"] = summary

        return report

# ==============================================================================
# 单元测试与对标 (对应论文: 拿最经典的 MAPbI3 钙钛矿材料做试金石)
# ==============================================================================
if __name__ == "__main__":
    # 确保你设置了 MP_API_KEY 环境变量
    pipeline = HighThroughputScreening()
    
    # 论文实战对标 1: 经典有机无机杂化钙钛矿
    target_material = "CH3NH3PbI3" 
    
    try:
        result = pipeline.run_evaluation(target_material)
        print("\n" + "="*50)
        print("📊 [最终研报] 自动化筛选输出结果")
        print("="*50)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n💬 供大模型读取的自然语言摘要:")
        print(result.get("LLM_Summary"))
        
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
