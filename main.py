import os
import sys

# 将项目根目录加入系统路径，确保所有 src 下的模块能被正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.orchestrator import MatMoEOrchestrator

def print_banner():
    """打印欢迎界面"""
    banner = """
============================================================
      🌞 Welcome to MatMoE-Agent 🌞
   (Photovoltaic Materials Mixture of Experts Agent)
============================================================
  Modules Loaded:
  - 🧠 LLM Brain (Qwen/Llama)
  - 📚 Knowledge Expert (RAG)
  - 🗄️ Database Expert (Materials Project)
  - ⚛️ Physics Expert (CHGNet & S-Q Limit)
============================================================
    """
    print(banner)

def main():
    print_banner()
    print("⏳ 正在初始化系统组件并加载模型，请稍候...")
    
    try:
        # 实例化总调度器
        agent = MatMoEOrchestrator()
    except Exception as e:
        print(f"\n❌ 系统初始化失败，请检查模型路径或 API 配置: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ MatMoE-Agent 启动成功！")
    print("💡 提示：输入 'exit' 或 'quit' 退出系统。")
    print("-" * 60)

    # 进入多轮对话的交互循环
    while True:
        try:
            # 接收用户输入
            user_input = input("\n🧪 User: ").strip()
            
            # 退出指令
            if user_input.lower() in ['exit', 'quit']:
                print("👋 感谢使用 MatMoE-Agent，再见！")
                break
            
            if not user_input:
                continue
            
            # 调用 Agent 核心逻辑
            response = agent.chat(user_input)
            
            # 打印最终回答
            print(f"\n🤖 MatMoE-Agent: \n{response}")
            print("-" * 60)

        except KeyboardInterrupt:
            # 捕获 Ctrl+C 强制退出
            print("\n\n👋 会话已中断。再见！")
            break
        except Exception as e:
            print(f"\n❌ 运行过程中发生未知错误: {e}")

if __name__ == "__main__":
    main()
