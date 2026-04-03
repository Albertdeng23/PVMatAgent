import os
import math
import yaml
from typing import Dict, Union, Tuple
from pymatgen.core.periodic_table import Element

class ThermoKineticExpert:
    """
    热力学与动力学深度分析专家。
    采用 Pymatgen 权威 Shannon 半径库与外部有机配置混合模式，计算钙钛矿容差因子。
    """
    
    def __init__(self):
        print("🔥 [ThermoKinetic_Expert] Initializing Thermodynamic & Kinetic module...")
        self.organic_radii = self._load_organic_radii()

    def _load_organic_radii(self) -> dict:
        """加载解耦的有机伪离子半径数据库"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        yaml_path = os.path.join(base_dir, "configs", "organic_radii.yaml")
        
        if not os.path.exists(yaml_path):
            print("⚠️ 警告: configs/organic_radii.yaml 未找到，仅支持纯无机钙钛矿计算。")
            return {}
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('organics', {})

    def _get_ionic_radius(self, ion_str: str, site_type: str) -> Tuple[float, str]:
        """
        核心查表逻辑：根据晶体学格位提取权威离子半径
        A位 (CN=12), B位 (CN=6), X位 (CN=6)
        返回: (半径值, 数据来源标注)
        """
        ion_str_upper = ion_str.upper()
        
        # 1. 优先检查是否为有机大阳离子 (仅允许出现在 A 位)
        if ion_str_upper in self.organic_radii:
            if site_type != 'A':
                raise ValueError(f"有机离子 {ion_str} 体积过大，只能占据 A 位，不能占据 {site_type} 位。")
            r = self.organic_radii[ion_str_upper]['radius_A']
            cit = self.organic_radii[ion_str_upper]['citation']
            return r, f"Organic Effective Radius ({cit})"

        # 2. 调用 Pymatgen 处理无机元素
        try:
            # 标准化大小写 (如 cs -> Cs, PB -> Pb)
            el = Element(ion_str.capitalize())
        except ValueError:
            raise ValueError(f"无法识别的元素或离子缩写: {ion_str}")

        # 3. 根据格位设定典型氧化态和配位数 (以卤化物钙钛矿 ABX3 为默认场景)
        try:
            if site_type == 'A':
                # A位通常是 +1 价，配位数 12
                radius = el.data["Shannon radii"]["1"]["XII"][""]["r_ionic"]
            elif site_type == 'B':
                # B位通常是 +2 价，配位数 6
                radius = el.data["Shannon radii"]["2"]["VI"][""]["r_ionic"]
            elif site_type == 'X':
                # X位通常是 -1 价，配位数 6
                radius = el.data["Shannon radii"]["-1"]["VI"][""]["r_ionic"]
            else:
                raise ValueError("未知的格位类型")
            return radius, "Shannon Radius (Pymatgen)"
            
        except KeyError:
            # 如果特定的氧化态/配位数在 Shannon 库中不存在（某些冷门元素）
            # 回退使用未指定环境的通用平均离子半径
            fallback_radius = el.average_ionic_radius
            if fallback_radius:
                return fallback_radius, "Average Ionic Radius (Pymatgen Fallback)"
            else:
                raise ValueError(f"无法在数据库中找到元素 {el.symbol} 的有效离子半径数据。")

    def calc_goldschmidt_tolerance(self, a_ion: str, b_ion: str, x_ion: str) -> Dict[str, Union[float, str, dict]]:
        """
        严谨版：计算 ABX3 卤化物钙钛矿的容差因子。
        """
        try:
            r_a, src_a = self._get_ionic_radius(a_ion, 'A')
            r_b, src_b = self._get_ionic_radius(b_ion, 'B')
            r_x, src_x = self._get_ionic_radius(x_ion, 'X')
        except ValueError as e:
            return {"error": str(e)}

        t = (r_a + r_x) / (math.sqrt(2) * (r_b + r_x))
        mu = r_b / r_x

        is_3d_perovskite = (0.80 <= t <= 1.05) and (0.44 <= mu <= 0.90)
        
        assessment = "具有形成稳定 3D 钙钛矿结构的极大潜力。"
        if t > 1.05: assessment = "A 位离子过大，极易形成 2D 钙钛矿或六方非钙钛矿相（如 1D/0D）。"
        elif t < 0.80: assessment = "A 位离子过小，晶格严重扭曲，倾向于形成正交相或非钙钛矿相。"
        if mu < 0.44: assessment += " 且 B 位离子过小，八面体骨架无法维持。"

        return {
            "target_system": f"{a_ion.capitalize()}{b_ion.capitalize()}{x_ion.capitalize()}3",
            "calculated_indices": {
                "tolerance_factor_t": round(t, 4),
                "octahedral_factor_mu": round(mu, 4),
                "is_potentially_stable_3D": is_3d_perovskite
            },
            "radii_data_used_Angstrom": {
                "A_site": {"ion": a_ion, "radius": r_a, "source": src_a},
                "B_site": {"ion": b_ion, "radius": r_b, "source": src_b},
                "X_site": {"ion": x_ion, "radius": r_x, "source": src_x}
            },
            "structural_assessment": assessment
        }

if __name__ == "__main__":
    expert = ThermoKineticExpert()
    
    # 1. 测试常规有机-无机杂化
    print("--- Testing MAPbI3 ---")
    print(expert.calc_goldschmidt_tolerance("MA", "Pb", "I"))

    # 2. 测试纯无机 (调用 Pymatgen Shannon 半径)
    print("\n--- Testing CsSnBr3 ---")
    print(expert.calc_goldschmidt_tolerance("Cs", "Sn", "Br"))

    # 3. 测试冷门元素或错误输入拦截能力
    print("\n--- Testing Error Handling (MA on B site) ---")
    print(expert.calc_goldschmidt_tolerance("Cs", "MA", "Cl"))
