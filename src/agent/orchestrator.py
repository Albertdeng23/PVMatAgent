import os
import sys
import json
import datetime
import warnings

# 确保项目根目录在 sys.path 中
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.append(base_dir)

warnings.filterwarnings("ignore")

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool

# ==============================================================================
# 导入所有底层专家模块
# ==============================================================================
from src.agent.brain import LLMBrain
from src.agent.prompts import PromptManager

# 1. 知识与数据库专家
from src.knowledge.rag_engine import KnowledgeExpert
from src.database.mp_handler import MPDatabaseExpert

# 2. 物理与底层计算专家
from src.physics.stability import CHGNetExpert
from src.physics.pv_calculators import PVCalculatorExpert
from src.physics.electronic_optics import ElectronicOpticsExpert
from src.physics.thermo_kinetics import ThermoKineticExpert

# 3. 高阶光伏物理专家 (NEW)
from src.physics.advanced_pv_slme import SLMEExpert
from src.physics.advanced_pv_tandem import TandemPVExpert

# 4. 工程、商业与实验专家
from src.engineering.techno_economic import TechnoEconomicExpert
from src.synthesis.lab_assistant import LabSynthesisExpert

# 5. 材料发现与晶体生成专家
from src.discovery.crystal_generator import CrystalGeneratorExpert


class MatMoEOrchestrator:
    """
    MatMoE-Agent 终极形态中枢调度器 (Tool Calling 架构)。
    统合检索、物理计算、材料发现、高阶光伏模拟等 13 个专家工具，并支持自动导出 Markdown 报告。
    """
    def __init__(self):
        print("🌐 [Orchestrator] Assembling Ultimate Tool-Calling Agent System...")
        
        self.base_dir = base_dir
        self.report_dir = os.path.join(self.base_dir, "logs", "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        
        self.brain = LLMBrain()
        self.llm = self.brain.get_llm()
        
        # ================= 实例化所有专家 =================
        self.rag_expert = KnowledgeExpert()
        self.mp_expert = MPDatabaseExpert()
        self.chgnet_expert = CHGNetExpert()
        self.pv_expert = PVCalculatorExpert()
        self.eo_expert = ElectronicOpticsExpert()
        self.tk_expert = ThermoKineticExpert()
        self.slme_expert = SLMEExpert()       # 新增：SLME专家
        self.tandem_expert = TandemPVExpert() # 新增：叠层匹配专家
        self.te_expert = TechnoEconomicExpert()
        self.lab_expert = LabSynthesisExpert()
        self.crystal_expert = CrystalGeneratorExpert()
        
        self.tools = self._register_tools()
        self.agent_executor = self._build_agent_executor()
        print(f"🚀 [Orchestrator] System Ready! {len(self.tools)} Super Tools mounted.")

    def _register_tools(self):
        """定义并注册供大模型调用的全部工具"""

        @tool
        def tool_rag_search(query: str) -> str:
            """【文献检索专家】用途：查询科学原理、未知的物理参数（如文献记录的真实带隙、直接带隙）、机制解释时调用。"""
            print(f"\n📚 [Agent Calls RAG]: {query}")
            return self.rag_expert.search(query)

        @tool
        def tool_structure_search(formula: str) -> str:
            """【结构下载专家】用途：从 MP 下载晶体结构 (CIF)。如果要计算形成能或预测带隙，必须先调此工具。"""
            print(f"\n🔍 [Agent Calls MP_DB]: {formula}")
            try:
                res = self.mp_expert.search_material(formula, limit=1)
                return res[0]['cif_path'] if res else f"未找到 {formula} 的 CIF 结构。"
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_generate_by_substitution(base_cif_path: str, substitution_mapping: str) -> str:
            """【元素替换新材料生成器】基于已有 CIF，将某些元素替换为新元素。输入: base_cif_path, substitution_mapping (如 "Pb:Sn, I:Br")。"""
            print(f"\n✨ [Agent Calls Substitution]: {os.path.basename(base_cif_path)} | {substitution_mapping}")
            try: return str(self.crystal_expert.generate_by_substitution(base_cif_path, substitution_mapping))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_generate_cubic_perovskite(a_ion: str, b_ion: str, x_ion: str) -> str:
            """【理想立方钙钛矿生成器】从零开始生成一个理想的 Pm-3m 无机 ABX3 钙钛矿初始晶体结构 (CIF)。"""
            print(f"\n✨ [Agent Calls Prototype Gen]: ABX3 ({a_ion}{b_ion}{x_ion}3)")
            try: return str(self.crystal_expert.generate_cubic_perovskite(a_ion, b_ion, x_ion))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_property_calculation(file_path: str, task: str = "formation") -> str:
            """【热力学稳定性专家 (CHGNet)】读取 CIF 文件，计算 formation(形成能), relax(弛豫)。用于对新结构进行优化或判断稳定性。"""
            print(f"\n⚛️ [Agent Calls CHGNet]: {os.path.basename(file_path)} | Task: {task}")
            try: return str(self.chgnet_expert.predict_properties(file_path, [task]))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_bandgap_predictor(file_path: str) -> str:
            """【高精度带隙预测专家 (MEGNet)】读取 CIF 文件，预测材料带隙 (Bandgap)。"""
            print(f"\n🌈 [Agent Calls MEGNet]: {os.path.basename(file_path)}")
            try: return str(self.eo_expert.predict_bandgap(file_path))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_electronic_band_analyzer(mp_id: str) -> str:
            """【能带结构与有效质量专家】获取导带底(CBM)和价带顶(VBM)能量级别。输入: mp_id"""
            print(f"\n📈 [Agent Calls Band Analyzer]: {mp_id}")
            try: return str(self.eo_expert.calc_effective_mass(mp_id))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_pv_performance_calc(bandgap_ev: float) -> str:
            """【S-Q 极限光伏效率专家】基础的光伏效率计算器。只考虑了黑体辐射，不考虑材料是否为间接带隙或厚度限制。"""
            print(f"\n☀️ [Agent Calls PV_Calc (S-Q)]: Bandgap = {bandgap_ev} eV")
            try: return str(self.pv_expert.calculate_sq_limit(float(bandgap_ev)))
            except Exception as e: return f"出错: {str(e)}"

        # ------------------- 高阶光伏物理 (NEW) -------------------
        @tool
        def tool_slme_efficiency_calculator(bandgap_ev: float, direct_bandgap_ev: float, thickness_um: float = 2.0) -> str:
            """
            【高阶光伏物理：SLME 光谱限制最大效率专家】
            用途：比 S-Q 极限更严谨的效率计算器。
            输入参数：
            1. bandgap_ev: 最小带隙 (基础带隙)。
            2. direct_bandgap_ev: 直接跃迁带隙。**警告：如果材料是间接带隙（如硅 Si），它的直接带隙必然远大于基础带隙（例如硅的基础带隙为1.12，直接带隙约为3.4）。调用前请务必通过 RAG 查清楚这两个值的区别，绝不可填成一样！**如果材料是直接带隙，则两者相等。
            3. thickness_um: 薄膜厚度(微米)。
            """
            print(f"\n🔬 [Agent Calls SLME]: Eg={bandgap_ev}, Eg_dir={direct_bandgap_ev}, L={thickness_um}um")
            try: return str(self.slme_expert.calculate_slme(float(bandgap_ev), float(direct_bandgap_ev), float(thickness_um)))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_tandem_current_matcher(bandgap_top_ev: float, bandgap_bot_ev: float) -> str:
            """
            【高阶光伏物理：两端(2T)叠层太阳能电池匹配专家】
            用途：评估两种材料串联组合的潜力。计算电流匹配度(Mismatch)，并给出带隙优化建议，突破单结 S-Q 极限。
            输入：bandgap_top_ev(顶电池宽带隙), bandgap_bot_ev(底电池窄带隙)。
            """
            print(f"\n🔋 [Agent Calls Tandem Matcher]: Top={bandgap_top_ev} eV, Bottom={bandgap_bot_ev} eV")
            try: return str(self.tandem_expert.calculate_2t_tandem(float(bandgap_top_ev), float(bandgap_bot_ev)))
            except Exception as e: return f"出错: {str(e)}"

        # ------------------- 特性快筛与商业评估 -------------------
        @tool
        def tool_goldschmidt_tolerance(a_ion: str, b_ion: str, x_ion: str) -> str:
            """【钙钛矿容差因子快筛专家】评估 ABX3 钙钛矿的 3D 成相潜力。输入：A, B, X 离子"""
            print(f"\n📐 [Agent Calls Tolerance]: A={a_ion}, B={b_ion}, X={x_ion}")
            try: return str(self.tk_expert.calc_goldschmidt_tolerance(a_ion, b_ion, x_ion))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_commercial_assessment(formula: str) -> str:
            """【商业成本与毒性环保专家】估算合成 1kg 材料的原料成本(USD)，筛查受限剧毒重金属。"""
            print(f"\n💰 [Agent Calls Techno-Economic]: {formula}")
            try: return str(self.te_expert.evaluate_material(formula))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_solvent_recommendation(precursor: str) -> str:
            """【实验合成与溶剂推荐专家】基于HSP推荐最佳的旋涂主溶剂和反溶剂组合工艺。"""
            print(f"\n🧪 [Agent Calls Synthesis Lab]: {precursor}")
            try: return str(self.lab_expert.recommend_solvent_system(precursor))
            except Exception as e: return f"出错: {str(e)}"

        return [
            tool_rag_search, tool_structure_search, 
            tool_generate_by_substitution, tool_generate_cubic_perovskite,
            tool_property_calculation, tool_bandgap_predictor, tool_electronic_band_analyzer, 
            tool_pv_performance_calc, tool_slme_efficiency_calculator, tool_tandem_current_matcher, # SLME 和 叠层
            tool_goldschmidt_tolerance, tool_commercial_assessment, tool_solvent_recommendation
        ]

    def _build_agent_executor(self):
        prompt = PromptManager.get_tool_calling_prompt()
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=25,
            return_intermediate_steps=True
        )

    def _save_to_markdown(self, user_query: str, final_output: str, steps: list):
        """将最终报告和思考调用轨迹固化为 Markdown 文件。"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"MatReport_{timestamp}.md"
        filepath = os.path.join(self.report_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 🔬 MatMoE-Agent 综合科研报告\n\n")
            f.write(f"> **生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"> **用户指令**: {user_query}\n\n")
            f.write(f"---\n\n")
            
            f.write(f"## 📑 最终研究报告 (Final Report)\n\n{final_output}\n\n")
            f.write(f"---\n\n")
            
            f.write(f"## 🧠 智能体工具调用轨迹 (Agent Trace)\n\n")
            
            if not steps:
                f.write("*本次响应未调用任何外部工具。*\n")
            else:
                for i, (action, observation) in enumerate(steps, 1):
                    f.write(f"### 步骤 {i}: `{action.tool}`\n")
                    f.write(f"**输入参数 (Action Input)**:\n```json\n")
                    try:
                        input_str = json.dumps(action.tool_input, indent=2, ensure_ascii=False)
                    except:
                        input_str = str(action.tool_input)
                    f.write(f"{input_str}\n```\n\n")
                    f.write(f"**工具返回结果 (Observation)**:\n```text\n")
                    f.write(f"{observation}\n```\n\n")

        print(f"\n📄 [System] 完整的 Markdown 科研报告已保存至: {filepath}")

    def chat(self, user_query: str):
        try:
            print(f"\n{'='*50}\n👤 User: {user_query}\n{'='*50}")
            response = self.agent_executor.invoke({"input": user_query})
            
            final_output = response.get("output", "")
            intermediate_steps = response.get("intermediate_steps", [])
            
            self._save_to_markdown(user_query, final_output, intermediate_steps)
            
            return final_output
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ 运行崩溃: {str(e)}"
