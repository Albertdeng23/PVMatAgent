import os
import yaml
import warnings
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 屏蔽 MP API 的警告
warnings.filterwarnings("ignore", category=UserWarning, module="mp_api")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

def load_config():
    """加载全局 YAML 配置文件获取缓存路径"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class MPDatabaseExpert:
    """
    Materials Project 数据库搜索专家。
    负责规范化化学式、检索最稳定结构，并下载 CIF 到统一缓存目录。
    """
    def __init__(self):
        print("🗄️ [MP_Expert] Initializing Materials Project interface...")
        load_dotenv(override=True)
        
        # 1. 获取 API Key
        self.api_key = os.getenv("MP_API_KEY")
        if not self.api_key:
            raise ValueError("❌ MP_API_KEY 未在 .env 文件中设置。请添加后重试。")
            
        # 2. 读取配置并设置缓存目录
        self.config = load_config()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_rel_path = self.config['system']['cache_dir']
        self.save_dir = os.path.join(self.base_dir, cache_rel_path.replace("./", ""))
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 3. 验证连接
        try:
            with MPRester(self.api_key) as mpr:
                pass
            print(f"✅ [MP_Expert] Connected to MP. Cache dir: {self.save_dir}")
        except Exception as e:
            print(f"❌ [MP_Expert] Connection failed: {e}")

    def _normalize_formula(self, query: str) -> str:
        """处理常见的钙钛矿缩写和下标符号"""
        abbreviations = {
            "MAPbI3": "CH3NH3PbI3", "FAPbI3": "CH5N2PbI3",
            "MAPbBr3": "CH3NH3PbBr3", "FAPbBr3": "CH5N2PbBr3",
            "MA": "CH3NH3", "FA": "CH5N2"
        }
        for abbr, full in abbreviations.items():
            if query.upper() == abbr.upper():
                query = full
                break

        subscript_map = {
            '₀':'0', '₁':'1', '₂':'2', '₃':'3', '₄':'4',
            '₅':'5', '₆':'6', '₇':'7', '₈':'8', '₉':'9'
        }
        for sub, num in subscript_map.items():
            query = query.replace(sub, num)
            
        # 清除可能带入的标点符号
        query = query.replace("formula=", "").strip('.,;!"\' ')
        return query

    def search_material(self, query: str, limit: int = 1) -> List[Dict]:
        """
        核心搜索接口：搜索最稳定的材料结构并保存 CIF 文件。
        供 Agent 作为工具调用。
        """
        query = self._normalize_formula(query)
        print(f"🔍 [MP_Expert] Searching for: '{query}'...")

        try:
            with MPRester(self.api_key) as mpr:
                # 支持 ID 搜索或化学式搜索
                if "mp-" in query or "mvc-" in query:
                    docs = mpr.summary.search(
                        material_ids=[query],
                        fields=["material_id", "formula_pretty", "structure", 
                                "energy_above_hull", "is_stable", "symmetry", "band_gap"]
                    )
                else:
                    docs = mpr.summary.search(
                        formula=query,
                        fields=["material_id", "formula_pretty", "structure", 
                                "energy_above_hull", "is_stable", "symmetry", "band_gap"]
                    )

            if not docs:
                return []

            # 智能排序：优先返回稳定材料，其次能量最低 (energy_above_hull 越小越好)
            sorted_docs = sorted(docs, key=lambda x: (not x.is_stable, x.energy_above_hull))
            top_docs = sorted_docs[:limit]

            results = []
            for i, doc in enumerate(top_docs):
                mat_id = str(doc.material_id)
                formula = doc.formula_pretty
                
                # 保存 CIF
                file_name = f"{formula}_{mat_id}.cif"
                file_path = os.path.join(self.save_dir, file_name)
                
                try:
                    doc.structure.to(filename=file_path, fmt="cif")
                except Exception as e:
                    print(f"⚠️ [MP_Expert] Write CIF failed for {mat_id}: {e}")
                    continue
                
                # 提取对称性（供物理属性分析）
                try:
                    sga = SpacegroupAnalyzer(doc.structure)
                    crystal_sys = sga.get_crystal_system()
                    space_group = sga.get_space_group_symbol()
                except:
                    crystal_sys, space_group = "Unknown", "Unknown"

                results.append({
                    "material_id": mat_id,
                    "formula": formula,
                    "cif_path": os.path.abspath(file_path),
                    "is_stable": doc.is_stable,
                    "crystal_system": crystal_sys,
                    "space_group": space_group,
                    "band_gap_eV": round(doc.band_gap, 3) if doc.band_gap else None
                })

            return results

        except Exception as e:
            print(f"❌ [MP_Expert] Search error: {e}")
            return []

if __name__ == "__main__":
    # 单元测试（请确保你的 .env 文件里有 MP_API_KEY=your_key）
    try:
        mp = MPDatabaseExpert()
        # 测试搜索钙钛矿常见缩写
        res = mp.search_material("MAPbI3", limit=1)
        if res:
            print("\n✅ Search Success:")
            print(f"Formula: {res[0]['formula']}")
            print(f"CIF Path: {res[0]['cif_path']}")
            print(f"Crystal System: {res[0]['crystal_system']}")
        else:
            print("⚠️ 未找到结果或网络异常。")
    except Exception as e:
        print(f"测试中断: {e}")
