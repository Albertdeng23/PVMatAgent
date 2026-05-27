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

# ==============================================================================
# Windows DLL 加载顺序修复：必须先加载科学计算模块 (PyMatGen/TensorFlow)，
# 再加载 LangChain 模块，避免 DLL 冲突导致段错误。
# ==============================================================================

# 1. 知识与数据库专家
from src.knowledge.rag_engine import KnowledgeExpert
from src.database.mp_handler import MPDatabaseExpert

# 2. 物理与底层计算专家
from src.physics.stability import CHGNetExpert
from src.physics.pv_calculators import PVCalculatorExpert
from src.physics.electronic_optics import ElectronicOpticsExpert
from src.physics.thermo_kinetics import ThermoKineticExpert

# 3. 高阶光伏物理专家
from src.physics.advanced_pv_slme import SLMEExpert
from src.physics.advanced_pv_tandem import TandemPVExpert

# 4. 工程、商业与实验专家
from src.engineering.techno_economic import TechnoEconomicExpert
from src.synthesis.lab_assistant import LabSynthesisExpert

# 5. 材料发现与晶体生成专家
from src.discovery.crystal_generator import CrystalGeneratorExpert

# 6. 第一性原理计算专家 (VASP)
from src.calculations.vasp_tools import VASPToolsExpert

# 7. LangChain 核心 (必须在科学模块之后导入)
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 8. Agent 大脑模块 (最后导入，依赖 LangChain)
from src.agent.brain import LLMBrain
from src.agent.prompts import PromptManager


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
        self.vasp_expert = VASPToolsExpert()  # 新增：VASP第一性原理计算专家
        
        self.tools = self._register_tools()
        self.agent_graph = self._build_agent_graph()
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
            """【结构下载专家】用途：从 MP 下载晶体结构 (CIF)。如果要计算形成能或预测带隙，必须先调此工具。返回 JSON 包含 cif_path、formula、material_id、crystal_system、mp_bandgap_eV、is_stable。"""
            print(f"\n🔍 [Agent Calls MP_DB]: {formula}")
            try:
                res = self.mp_expert.search_material(formula, limit=1)
                if not res:
                    return json.dumps({
                        "error": f"Materials Project 中未找到与 '{formula}' 匹配的结构。",
                        "suggestion": "请尝试用更简化的化学式（如去掉有机基团），或调用 tool_rag_search 查阅文献中的结构信息，或用 tool_generate_cubic_perovskite 从零构建。"
                    }, ensure_ascii=False)
                r = res[0]
                return json.dumps({
                    "cif_path": r["cif_path"],
                    "formula": r["formula"],
                    "material_id": r["material_id"],
                    "crystal_system": r.get("crystal_system", "Unknown"),
                    "mp_bandgap_eV": r.get("band_gap_eV"),
                    "mp_is_stable": r.get("is_stable"),
                }, ensure_ascii=False)
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_generate_by_substitution(base_cif_path: str, substitution_mapping: str) -> str:
            """【元素替换新材料生成器】基于已有 CIF，将某些元素全替换为新元素。支持有机基团（MA/FA/EA/GA）。输入: base_cif_path, substitution_mapping (如 "Pb:Sn, I:Br" 或 "I:Br")。若需部分替换（如仅替换50%的I），改用 tool_generate_partial_substitution。"""
            print(f"\n✨ [Agent Calls Substitution]: {os.path.basename(base_cif_path)} | {substitution_mapping}")
            try: return str(self.crystal_expert.generate_by_substitution(base_cif_path, substitution_mapping))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_generate_cubic_perovskite(a_ion: str, b_ion: str, x_ion: str) -> str:
            """【理想立方钙钛矿生成器】从零开始生成一个理想的 Pm-3m ABX3 钙钛矿初始晶体结构 (CIF)。支持有机A位（MA/FA/EA/GA）和无机A/B/X位。生成的CIF可直接用于后续弛豫和带隙计算。"""
            print(f"\n✨ [Agent Calls Prototype Gen]: ABX3 ({a_ion}{b_ion}{x_ion}3)")
            try: return str(self.crystal_expert.generate_cubic_perovskite(a_ion, b_ion, x_ion))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_generate_double_perovskite(a_ion: str, b_ion: str, b_prime_ion: str, x_ion: str) -> str:
            """【双钙钛矿生成器】从零生成岩盐有序的立方双钙钛矿 A2BB'X6 (Fm-3m)。用于探索无铅双钙钛矿候选材料。输入: A位, B位, B'位, X位 四种离子。"""
            print(f"\n✨ [Agent Calls Double Pv]: A2BB'X6 ({a_ion}2{b_ion}{b_prime_ion}{x_ion}6)")
            try: return str(self.crystal_expert.generate_double_perovskite(a_ion, b_ion, b_prime_ion, x_ion))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_generate_partial_substitution(base_cif_path: str, substitution_spec: str) -> str:
            """【部分替换生成器】基于已有CIF，将一定比例的某元素替换为另一元素，生成中间组分。输入: base_cif_path, substitution_spec (格式: "old_element->new_element, fraction=0.XX"，如 "I->Br, fraction=0.15" 表示15%的I被Br替换)。**重要**：如需生成 CsSn(I0.5Br0.5)3 等中间组分，用此工具而非全替换。"""
            print(f"\n✨ [Agent Calls Partial Sub]: {os.path.basename(base_cif_path)} | {substitution_spec}")
            try: return str(self.crystal_expert.generate_partial_substitution(base_cif_path, substitution_spec))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_generate_from_template(template: str, ions_json: str) -> str:
            """【模板化结构生成器】根据结构模板和离子JSON列表生成CIF。用于当MP数据库中找不到某材料时从零构建。template可选: 'cubic'(ABX3), 'double_perovskite'(A2BB'X6), 'tetragonal'(ABX3四方相)。ions_json格式: '["A离子","B离子","X离子"]' 或双钙钛矿的 '["A","B","B'","X"]'。"""
            print(f"\n✨ [Agent Calls Template Gen]: {template} | {ions_json}")
            try:
                import json as _json
                ions = _json.loads(ions_json)
                return str(self.crystal_expert.generate_from_template(template, ions))
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

        # ------------------- 第一性原理计算专家 (VASP) -------------------
        @tool
        def tool_vasp_connect(hostname: str, username: str, password: str = "", 
                             key_filename: str = "", vasp_command: str = "vasp_std") -> str:
            """
            【第一性原理计算专家 (VASP) - 连接服务器】
            用途：连接到远程VASP计算服务器（超算中心或课题组服务器）。
            ⚠️ 警告：此工具仅用于连接服务器，不执行计算。正式计算请使用submit工具。
            输入：hostname(服务器地址), username(用户名), password(密码) 或 key_filename(密钥路径)。
            """
            print(f"\n⚛️ [Agent Calls VASP Connect]: {hostname}")
            try:
                kwargs = {"hostname": hostname, "username": username, "vasp_command": vasp_command}
                if password:
                    kwargs["password"] = password
                elif key_filename:
                    kwargs["key_filename"] = key_filename
                return str(self.vasp_expert.connect_server(**kwargs))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_vasp_prepare_and_submit(cif_path: str, calculation_type: str = "relax",
                                        xc: str = "PBE", kpts: str = "(4,4,4)", 
                                        encut: int = 400, blocking: bool = False) -> str:
            """
            【第一性原理计算专家 (VASP) - 提交计算】
            用途：准备VASP输入文件并提交第一性原理计算任务。
            
            ⚠️⚠️⚠️ 极其重要警告：
            1. VASP计算极其耗时（通常数小时到数天）且消耗大量计算资源！
            2. 请仅在以下情况才使用此工具：
               - 用户明确要求进行DFT计算
               - 机器学习预测结果存疑需要验证
               - 需要高精度能带结构数据
            3. 优先使用CHGNet/MEGNet等快速工具进行初步筛选！
            4. 提交前必须确保已连接服务器 (tool_vasp_connect)。
            
            输入：
            - cif_path: CIF文件路径
            - calculation_type: 计算类型 ('relax'-结构优化, 'static'-静态计算, 'band'-能带计算)
            - xc: 交换关联泛函 (默认PBE)
            - kpts: K点网格，字符串格式如"(4,4,4)"
            - encut: 截断能 (默认400 eV)
            - blocking: 是否阻塞等待完成 (默认False，推荐后台运行)
            """
            print(f"\n⚛️ [Agent Calls VASP Submit]: {os.path.basename(cif_path)} | {calculation_type}")
            try:
                # 解析kpts字符串
                import ast
                kpts_tuple = ast.literal_eval(kpts)
                
                # 准备输入文件
                from src.calculations.vasp_tools import VASPConfig
                config = VASPConfig(xc=xc, kpts=kpts_tuple, encut=encut)
                prep_result = self.vasp_expert.prepare_vasp_inputs(cif_path, config, calculation_type)
                
                if "error" in prep_result:
                    return str(prep_result)
                
                job_name = prep_result["job_name"]
                
                # 提交计算
                submit_result = self.vasp_expert.submit_calculation(job_name, blocking=blocking)
                
                return {
                    "preparation": prep_result,
                    "submission": submit_result,
                    "warning": "VASP计算已提交到远程服务器，可能需要数小时完成。使用tool_vasp_check_status检查进度。"
                }
            except Exception as e: 
                return f"出错: {str(e)}"

        @tool
        def tool_vasp_check_status(job_name: str = "") -> str:
            """
            【第一性原理计算专家 (VASP) - 检查状态】
            用途：检查VASP计算任务的状态。
            输入：job_name(任务名称，如省略则检查最新提交的任务)
            """
            print(f"\n⚛️ [Agent Calls VASP Status]: {job_name if job_name else 'latest'}")
            try:
                if not job_name:
                    # 获取最新任务名称
                    job_name = self.vasp_expert.current_job_id
                    if not job_name:
                        return {"error": "没有正在运行的任务，请提供job_name参数"}
                return str(self.vasp_expert.check_status(job_name))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_vasp_download_results(job_name: str = "") -> str:
            """
            【第一性原理专家 (VASP) - 下载结果】
            用途：下载已完成的VASP计算结果文件。
            输入：job_name(任务名称)
            """
            print(f"\n⚛️ [Agent Calls VASP Download]: {job_name if job_name else 'latest'}")
            try:
                if not job_name:
                    return {"error": "请提供job_name参数"}
                return str(self.vasp_expert.download_results(job_name))
            except Exception as e: return f"出错: {str(e)}"

        @tool
        def tool_vasp_parse_bandstructure(job_name: str) -> str:
            """
            【第一性原理计算专家 (VASP) - 解析能带】
            用途：解析VASP计算生成的EIGENVAL文件，提取能带结构数据。
            输入：job_name(任务名称)
            """
            print(f"\n⚛️ [Agent Calls VASP Band Analysis]: {job_name}")
            try:
                return str(self.vasp_expert.parse_eigenval(job_name))
            except Exception as e: return f"出错: {str(e)}"

        return [
            tool_rag_search, tool_structure_search,
            tool_generate_by_substitution, tool_generate_cubic_perovskite,
            tool_generate_double_perovskite, tool_generate_partial_substitution,  # 新增：双钙钛矿 + 部分替换
            tool_generate_from_template,                                           # 新增：模板化结构生成
            tool_property_calculation, tool_bandgap_predictor, tool_electronic_band_analyzer,
            tool_pv_performance_calc, tool_slme_efficiency_calculator, tool_tandem_current_matcher,
            tool_goldschmidt_tolerance, tool_commercial_assessment, tool_solvent_recommendation,
            tool_vasp_connect, tool_vasp_prepare_and_submit, tool_vasp_check_status,
            tool_vasp_download_results, tool_vasp_parse_bandstructure
        ]

    def _build_agent_graph(self):
        system_prompt = PromptManager.get_tool_calling_prompt()
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=True
        )

    def _extract_steps(self, messages):
        steps = []
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})
                    observation = ""
                    for j in range(i + 1, len(messages)):
                        mj = messages[j]
                        if isinstance(mj, ToolMessage) and mj.tool_call_id == tc.get("id"):
                            observation = mj.content
                            break
                    steps.append((tool_name, tool_args, observation))
        return steps

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
                for i, (tool_name, tool_args, observation) in enumerate(steps, 1):
                    f.write(f"### 步骤 {i}: `{tool_name}`\n")
                    f.write(f"**输入参数 (Action Input)**:\n```json\n")
                    try:
                        input_str = json.dumps(tool_args, indent=2, ensure_ascii=False)
                    except:
                        input_str = str(tool_args)
                    f.write(f"{input_str}\n```\n\n")
                    f.write(f"**工具返回结果 (Observation)**:\n```text\n")
                    f.write(f"{observation}\n```\n\n")

        print(f"\n📄 [System] 完整的 Markdown 科研报告已保存至: {filepath}")

    def chat(self, user_query: str):
        try:
            print(f"\n{'='*50}\n👤 User: {user_query}\n{'='*50}")
            config = {"recursion_limit": 50}
            response = self.agent_graph.invoke(
                {"messages": [HumanMessage(content=user_query)]},
                config=config
            )

            messages = response.get("messages", [])

            final_output = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    final_output = msg.content
                    break

            if not final_output and messages:
                final_output = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

            intermediate_steps = self._extract_steps(messages)

            self._save_to_markdown(user_query, final_output, intermediate_steps)

            return final_output
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ 运行崩溃: {str(e)}"
