import os
import yaml
import torch
import importlib.metadata
import warnings
from dotenv import load_dotenv

# 本地模型依赖
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

# 云端模型依赖
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

def load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class LLMBrain:
    """
    MatMoE-Agent 混合模型底座。
    支持一键切换本地 4bit 量化模型或云端 API (兼容 OpenAI 格式)。
    """
    def __init__(self):
        print("🧠 [Brain] Initializing LLM Engine...")
        load_dotenv(override=True)
        self.config = load_config()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.model_type = self.config['model'].get('type', 'local').lower()
        
        # 强制停止词，防止模型产生幻觉代替工具输出
        self.stop_words = ["OBSERVATION:", "Observation:", "OBSERVATION", "观察:"]

        if self.model_type == 'cloud':
            self.llm = self._load_cloud_model()
        else:
            self._apply_windows_patches()
            self.llm = self._load_local_model()

    def _apply_windows_patches(self):
        """仅在本地模型时生效：解决 Windows 下的 BitsAndBytes 问题"""
        bnb_path = r"C:\Users\80634\anaconda3\envs\Gpaper\Lib\site-packages\bitsandbytes"
        os.environ["PATH"] = bnb_path + ";" + os.environ.get("PATH", "")
        if hasattr(os, 'add_dll_directory'):
            try: os.add_dll_directory(bnb_path)
            except Exception: pass
        try:
            _orig_version = importlib.metadata.version
            importlib.metadata.version = lambda pkg: "0.43.2" if pkg == "bitsandbytes" else _orig_version(pkg)
        except Exception: pass

    def _load_local_model(self):
        """加载本地 4bit 量化模型"""
        model_path = os.path.join(self.base_dir, self.config['model']['llm_path'].replace("./", ""))
        print(f"   -> Loading LOCAL model from: {model_path}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=self.config['model']['max_new_tokens'],
            temperature=self.config['model']['temperature'],
            do_sample=True,
            repetition_penalty=self.config['model']['repetition_penalty'],
            return_full_text=False, pad_token_id=tokenizer.eos_token_id
        )
        
        # HuggingFace 的 stop 词需在 kwargs 传入
        return HuggingFacePipeline(pipeline=pipe, model_kwargs={"stop": self.stop_words})

    def _load_cloud_model(self):
        """加载云端 API 模型 (兼容 OpenAI 格式)"""
        cloud_cfg = self.config['model']['cloud_api']
        api_key = os.getenv("LLM_API_KEY") # 从 .env 中读取 API Key
        
        if not api_key:
            raise ValueError("❌ 采用云端模型时，必须在根目录 .env 文件中配置 LLM_API_KEY=xxx")

        print(f"   -> Loading CLOUD model: {cloud_cfg['model_name']} via {cloud_cfg['base_url']}")
        
        # 初始化 ChatOpenAI
        chat_model = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=cloud_cfg['base_url'],
            model_name=cloud_cfg['model_name'],
            temperature=self.config['model']['temperature'],
            max_tokens=self.config['model']['max_new_tokens']
        )
        
        # 云端模型的 stop 词通过 bind 方法绑定
        return chat_model.bind(stop=self.stop_words)

    def get_llm(self):
        """供 Orchestrator 调用的统一接口"""
        return self.llm

if __name__ == "__main__":
    # 测试
    brain = LLMBrain()
    llm = brain.get_llm()
    print("✅ [Brain] LLM successfully loaded. Testing simple generation...")
    print(llm.invoke("光伏材料的带隙通常是多少？"))
