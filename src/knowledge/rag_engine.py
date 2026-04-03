import os
import yaml
import torch
import warnings

# 屏蔽检索库的警告
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def load_config():
    """加载全局 YAML 配置文件获取 RAG 参数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class KnowledgeExpert:
    """
    领域知识检索专家 (RAG Engine)。
    实现 向量检索 + 关键词(BM25) 双路召回，并使用 Cross-Encoder 进行语义重排。
    用于回答大模型不懂的深层物理机制和文献数据。
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📚 [Knowledge_Expert] Initializing RAG Engine on {self.device}...")
        
        # 1. 加载配置
        self.config = load_config()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        rag_cfg = self.config.get('rag', {})
        db_rel_path = rag_cfg.get('vector_db_path', './data/vector_db')
        self.db_path = os.path.join(self.base_dir, db_rel_path.replace("./", ""))
        
        self.embed_model_name = rag_cfg.get('embedding_model', 'BAAI/bge-m3')
        self.rerank_model_name = rag_cfg.get('rerank_model', 'BAAI/bge-reranker-v2-m3')
        self.recall_k = rag_cfg.get('recall_k', 30)
        self.top_k = rag_cfg.get('top_k', 5)

        # 2. 初始化 Embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs={'device': self.device}
        )

        # 3. 连接向量数据库 (Chroma)
        if not os.path.exists(self.db_path):
            print(f"⚠️ [Knowledge_Expert] 向量数据库未找到 ({self.db_path})。请先运行预处理脚本构建数据库。")
            self.vectordb = None
            self.retrieval_chain = None
        else:
            self.vectordb = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            # 4. 构建检索管道
            self.retrieval_chain = self._build_retrieval_chain()
            print("✅ [Knowledge_Expert] RAG Pipeline Ready.")

    def _build_retrieval_chain(self):
        """构建 混合检索 + 重排 的完整链条"""
        try:
            # A. 向量检索器
            vector_retriever = self.vectordb.as_retriever(search_kwargs={"k": self.recall_k})

            # B. BM25 关键词检索器
            all_data = self.vectordb.get()
            if len(all_data['ids']) > 0:
                bm25_docs = [
                    Document(page_content=txt, metadata=meta) 
                    for txt, meta in zip(all_data['documents'], all_data['metadatas'])
                ]
                bm25_retriever = BM25Retriever.from_documents(bm25_docs)
                bm25_retriever.k = self.recall_k
                
                # 混合检索 (权重 4:6)
                base_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[0.4, 0.6] 
                )
            else:
                base_retriever = vector_retriever

            # C. 重排序模型
            reranker_model = HuggingFaceCrossEncoder(
                model_name=self.rerank_model_name,
                model_kwargs={'device': self.device}
            )
            compressor = CrossEncoderReranker(model=reranker_model, top_n=self.top_k)
            
            # D. 组合
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        except Exception as e:
            print(f"❌ [Knowledge_Expert] Failed to build pipeline: {e}")
            return None

    def _format_docs(self, docs) -> str:
        """格式化文档，便于 LLM 阅读"""
        if not docs:
            return "未检索到相关文献信息。"
            
        formatted_chunks = []
        for i, doc in enumerate(docs):
            src = os.path.basename(doc.metadata.get('source', 'Unknown Paper'))
            content = doc.page_content.strip()
            
            chunk_str = f"【文献 {i+1}】(来源: {src})\n{content}"
            formatted_chunks.append(chunk_str)
            
        return "\n\n" + "-"*40 + "\n\n".join(formatted_chunks) + "\n" + "-"*40

    def search(self, query: str) -> str:
        """
        核心搜索接口：供 Agent 作为工具调用。
        """
        if not self.retrieval_chain:
            return "错误：文献数据库未初始化，无法进行知识检索。"
            
        if not query:
            return "错误：检索词为空。"

        try:
            print(f"   -> [RAG Searching] {query}")
            relevant_docs = self.retrieval_chain.invoke(query)
            return self._format_docs(relevant_docs)
        except Exception as e:
            return f"检索过程中发生错误: {str(e)}"

if __name__ == "__main__":
    # 单元测试 (如果当前没有向量数据库，它会安全地给出警告而不会崩溃)
    expert = KnowledgeExpert()
    res = expert.search("钙钛矿太阳能电池的稳定性挑战是什么？")
    print(res)
