import os
import sys
import re
import warnings
import torch
from typing import Union

# --- LangChain & Transformers ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# --- 导入底层专家模块 (假设你在项目根目录运行) ---
try:
    from RAGsearch import SolarRAGSearch # 兼容你目前的命名
    from src.workflows.screening_pipeline import HighThroughputScreening
except ImportError as e:
    print(f"⚠️ 专家模块导入失败: {e}\n请确保在 MatMoE_Agent 根目录下运行，或检查环境变量 PYTHONPATH。")

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 强力输出解析器 (专门克制 8B 模型的“自问自答”幻觉)
# ==============================================================================
class MatMoEOutputParser(ReActSingleInputOutputParser):
    """
    定制化解析器：
    专门拦截 8B 模型喜欢替工具伪造 "OBSERVATION:" 输出的坏习惯。
    一刀切断所有伪造结果，强制交还控制权给 Python 执行环境。
    """
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        text = text.replace("</s>", "").replace("<|endoftext|>", "").replace("<|eot_id|>", "")
        
        # 核心拦截逻辑：截断幻觉
        for stop_word in ["OBSERVATION:", "Observation:", "Observation\n"]:
            if stop_word in text:
                text = text.split(stop_word)[0].strip()
                break 
        
        # 尝试匹配 Action (调用工具)
        action_match = re.search(r"Action:\s*(.*?)\n+Action Input:\s*(.*)", text, re.DOTALL)
        if action_match:
            return super().parse(text)
            
        # 尝试匹配 Final Answer (最终回复)
        if "Final Answer:" in text:
            ans = text.split("Final Answer:")[-1].strip()
            return AgentFinish(return_values={"output": ans}, log=text)
            
        return super().parse(text)

# ==============================================================================
# 2. 智能体中枢引擎
# ==============================================================================
class MatMoEAgent:
    def __init__(self, model_path: str):
        print("🧠 [MatMoE-Agent] 正在启动中枢神经系统...")
        self.model_path = model_path
        
        # 1. 挂载专家模块 (延迟加载/按需加载以节省显存)
        self._init_experts()
        
        # 2. 加载微调后的认知大脑 (LLM)
        self.llm = self._load_cognitive_brain()
        
        # 3. 组装多智能体工作流
        self.agent_executor = self._build_agent_workflow()
        print("✅ [MatMoE-Agent] 核心架构装配完毕，等待指令。")

    def _init_experts(self):
        print("   -> 挂载 [文献与理论检索专家] (RAG)...")
        # 如果你不想每次启动都消耗内存，可以在这里做懒加载，目前为了稳定直接实例化
        self.rag_expert = SolarRAGSearch() 
        
        print("   -> 挂载 [光伏物理评估专家] (Workflow)...")
        self.pv_expert = HighThroughputScreening()

    def _load_cognitive_brain(self):
        print(f"   -> 唤醒认知大脑 (加载 4bit 量化模型 {os.path.basename(self.model_path)})...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.01, # 极低温度保证逻辑严密性
            do_sample=False,  # 禁用随机采样，让工具调用更稳定
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        # 显式截断符
        return HuggingFacePipeline(pipeline=pipe, model_kwargs={"stop": ["OBSERVATION:", "Observation:"]})

    def _build_agent_workflow(self):
        
        # ==================== 工具定义区 ====================
        @tool
        def tool_search_literature(query: str) -> str:
            """
            【文献与理论检索专家】
            用途：用于回答光伏领域的科学原理、机制解释（如“为什么...”、“...有何影响”）、以及无法直接计算的物理性质（如载流子迁移率、光学吸收系数）。
            输入：具体的自然语言科学问题。
            """
            query = query.split('\n')[0].strip()
            print(f"\n📚 [Agent 路由 -> RAG 专家] 检索知识库: {query}")
            return self.rag_expert.search(query)

        @tool
        def tool_evaluate_pv_material(formula: str) -> str:
            """
            【光伏潜力自动化评估专家】
            用途：当用户要求“评估”、“筛选”或询问某个特定材料（如钙钛矿、硅）的“光电转换效率 (PCE)”、“开路电压 (Voc)”、“稳定性/形成能”时调用。
            输入：**必须是纯粹的化学式**，例如 "CH3NH3PbI3" 或 "Si" 或 "GaAs"。严禁输入其他文字。
            """
            formula = formula.split('\n')[0].replace("formula=", "").strip().rstrip(".,;!\"'")
            print(f"\n⚙️ [Agent 路由 -> 物理评估流水线] 启动全自动分析: {formula}")
            
            # 直接调用写好的流水线，避免大模型中途出错
            report = self.pv_expert.run_evaluation(formula)
            return report.get("LLM_Summary", str(report))

        tools = [tool_search_literature, tool_evaluate_pv_material]

        # ==================== 神经-符号系统 Prompt ====================
        template = """你是一个世界顶尖的光伏材料科学家，你的名字是 MatMoE-Agent。
你的大脑中注入了 3000 篇光伏前沿文献的知识，同时你还可以调用底层的物理计算引擎。

工具箱 (Tools):
------
{tools}

推理策略 (STRATEGY):
请仔细判断用户的意图，选择唯一正确的解决路径：
- 路径 A【理论求证】：如果用户问“为什么”、“原理”、“机理”或“迁移率”等复杂性质。-> 必须使用 `tool_search_literature`
- 路径 B【定量评估】：如果用户问某个具体材料的“效率”、“能做太阳能电池吗”、“稳定性如何”。-> 必须提取化学式，并使用 `tool_evaluate_pv_material`

严格的输出格式 (FORMAT):
```
Thought: 用户问的是[...], 属于路径[A/B]。我需要使用工具吗？Yes
Action: 工具名称 (只能是 [{tool_names}] 之一)
Action Input: 传入工具的参数 (纯粹的问题 或 纯粹的化学式)
```

OBSERVATION: 
(等待系统返回结果)

```
Thought: 我已经获得了足够的信息。我需要使用工具吗？No
Final Answer: [用专业的科研口吻，结合文献和物理计算数据，给出最终的详细解答。如果是定量评估，请列出效率、带隙、稳定性等具体指标。]
```

规则:
1. 在输出 "Action Input: xxx" 后必须**立即停止**生成，绝对不要自己伪造 OBSERVATION。
2. 对于光伏材料的效率预测，绝不能瞎编，必须依赖工具计算的结果。

开始执行！

用户输入: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(self.llm, tools, prompt, output_parser=MatMoEOutputParser())
        
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, # 打印思考过程
            handle_parsing_errors=True,
            max_iterations=5, # 减少迭代次数，避免无限死循环
            early_stopping_method="generate"
        )

    def chat(self, user_query: str):
        print(f"\n==================================================")
        print(f"🧑‍🔬 用户: {user_query}")
        print(f"==================================================")
        try:
            response = self.agent_executor.invoke({"input": user_query})
            final_answer = response["output"]
            print(f"\n🤖 MatMoE-Agent 总结回复:\n{final_answer}")
            return final_answer
        except Exception as e:
            return f"❌ 运行崩溃: {str(e)}"

# ==============================================================================
# 启动测试模块
# ==============================================================================
if __name__ == "__main__":
    # 配置你的模型路径 (使用之前你提供的路径)
    LOCAL_MODEL_PATH = r"C:\Users\80634\anaconda3\envs\Gpaper\Lib\site-packages\..." 
    # 注意：此处替换为实际的模型目录
    LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model", "merged_qwen_cpt_sft_CSimPO_V2.0")

    # 注意：为了让 BitsAndBytes 在 Windows 上正常运行，请确保你之前的补丁环境已经设置
    # (如果遇到 BNB 错误，请把之前 LLMQA.py 开头的补丁代码复制到这里最顶部)

    agent = MatMoEAgent(model_path=LOCAL_MODEL_PATH)
    
    # 对标测试 1: 走 RAG 路径的科学追问
    query_1 = "为什么相较于传统的硅电池，钙钛矿材料具有更高的光吸收系数？"
    agent.chat(query_1)
    
    # 对标测试 2: 走 Workflow 物理计算路径的实战评估
    query_2 = "帮我评估一下经典有机无机杂化材料 CH3NH3PbI3 的光伏潜力，它的极限效率能达到多少？"
    agent.chat(query_2)