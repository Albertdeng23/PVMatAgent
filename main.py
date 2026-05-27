import os
import sys

# Windows 下强制 UTF-8 编码，避免 emoji 字符导致 GBK 编码错误
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 将项目根目录加入系统路径，确保所有 src 下的模块能被正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

REQUIRED_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}

def _model_exists(model_name: str) -> bool:
    """检查模型目录是否存在且包含关键模型文件。"""
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isdir(model_path):
        return False
    # 检查目录中是否至少有 config.json 和模型权重文件
    has_config = os.path.isfile(os.path.join(model_path, "config.json"))
    has_weights = (
        os.path.isfile(os.path.join(model_path, "pytorch_model.bin")) or
        os.path.isfile(os.path.join(model_path, "model.safetensors"))
    )
    return has_config and has_weights

def ensure_models():
    """检查并自动下载缺失的嵌入/重排序模型。"""
    for local_name, repo_id in REQUIRED_MODELS.items():
        if _model_exists(local_name):
            print(f"✅ [Model Check] {local_name} 已就绪。")
            continue
        print(f"⬇️  [Model Check] {local_name} 未找到，正在从 HuggingFace 下载 {repo_id} ...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        target_dir = os.path.join(MODEL_DIR, local_name)
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ [Model Check] {local_name} 下载完成 -> {target_dir}")

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
    print("⏳ 正在检查本地模型...")
    ensure_models()
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
