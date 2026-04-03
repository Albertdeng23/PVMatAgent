import re
from typing import Union
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish

class MatMoEOutputParser(ReActSingleInputOutputParser):
    """
    MatMoE-Agent 专属的强力输出解析器 (Output Parser)。
    
    核心作用：
    拦截并截断 8B 轻量化模型在 ReAct 格式下常见的“自导自演”幻觉现象。
    
    处理逻辑优先级：
    1. 截断清洗：一旦发现模型输出了 "OBSERVATION:" 及其变体，立刻砍掉后面的所有内容。
    2. 工具调用判定 (Action)：如果清洗后包含 Action 格式，则强制解析为工具调用。
    3. 最终回复判定 (Final Answer)：如果没有 Action，但包含 Final Answer，则结束思考循环。
    4. 兜底解析：交由 LangChain 默认解析器处理。
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # 0. 基础清洗：移除模型可能生成的特殊结束符
        text = text.replace("</s>", "").replace("<|endoftext|>", "").replace("<|im_end|>", "")
        
        # 1. 【核心修复】强力截断幻觉
        # 如果模型自己写了 Observation，说明它在编造工具的运行结果。
        # 必须在这里一刀切断，把真正的执行权交还给 Python 代码。
        stop_words = [
            "OBSERVATION:", "Observation:", "Observation", "OBSERVATION", 
            "观察:", "观察结果:"
        ]
        for stop_word in stop_words:
            if stop_word in text:
                text = text.split(stop_word)[0].strip()
                break  # 只要命中一次截断即可
        
        # 2. 检查截断后是否包含有效的工具调用 (Action)
        # 正则匹配标准的 ReAct 格式：Action: [ToolName] \n Action Input: [Params]
        action_match = re.search(r"Action:\s*(.*?)\n+Action Input:\s*(.*)", text, re.DOTALL)
        
        if action_match:
            # 说明模型正确地输出了想要调用的工具和参数
            # 我们调用父类（ReActSingleInputOutputParser）的方法将其转换为 LangChain 标准的 AgentAction 对象
            return super().parse(text)
            
        # 3. 检查是否是最终结论 (Final Answer)
        if "Final Answer:" in text:
            # 提取 Final Answer 后面的纯文本作为最终输出给用户的内容
            final_text = text.split("Final Answer:")[-1].strip()
            return AgentFinish(
                return_values={"output": final_text},
                log=text, # 保留原始日志用于调试
            )
            
        # 4. 异常兜底
        # 既没有 Action 也没有 Final Answer，可能是模型卡在中间，尝试默认解析
        # 如果依然解析失败，LangChain 会在 Orchestrator 层面捕获异常并提示模型重新输出
        try:
            return super().parse(text)
        except Exception:
            # 如果彻底无法解析，强制中止并返回模型当前的原始输出，避免死循环报错
            return AgentFinish(
                return_values={"output": text.strip()},
                log=text,
            )


if __name__ == "__main__":
    # ================= 单元测试 =================
    print("🧪 [Parser] Running tests...")
    parser = MatMoEOutputParser()

    # 测试案例 1：标准工具调用（附带幻觉）
    test_text_1 = """Thought: 我需要查一下资料。
Action: tool_rag_search
Action Input: 钙钛矿的带隙是多少？
OBSERVATION: 钙钛矿的带隙通常在 1.5 eV 左右。
Thought: 我知道了。
Final Answer: 是 1.5 eV。"""

    res1 = parser.parse(test_text_1)
    print("\n--- Test 1 (Action with Hallucination) ---")
    print(f"Type: {type(res1)}")
    if isinstance(res1, AgentAction):
        print(f"Tool: {res1.tool}")
        print(f"Tool Input: {res1.tool_input}")

    # 测试案例 2：标准最终回复
    test_text_2 = """Thought: 我已经计算完毕，可以回答用户了。
Final Answer: 该材料的理论极限 PCE 为 25%。"""

    res2 = parser.parse(test_text_2)
    print("\n--- Test 2 (Final Answer) ---")
    print(f"Type: {type(res2)}")
    if isinstance(res2, AgentFinish):
        print(f"Output: {res2.return_values['output']}")
