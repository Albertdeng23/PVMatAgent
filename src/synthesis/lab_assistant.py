import os
import math
import yaml
from typing import Dict, List, Union

class LabSynthesisExpert:
    """
    实验合成助手专家。
    功能：
    1. 基于汉森溶解度参数 (HSP) 计算前驱体在特定溶剂中的溶解度 (Ra 距离)。
    2. 自动推荐最适合的主溶剂-反溶剂组合，指导旋涂工艺。
    """
    def __init__(self):
        print("🧪 [Synthesis_Expert] Initializing Laboratory Assistant module...")
        self.db = self._load_database()

    def _load_database(self) -> dict:
        """加载解耦的汉森溶解度参数数据库"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        yaml_path = os.path.join(base_dir, "configs", "hsp_database.yaml")
        
        if not os.path.exists(yaml_path):
            print("⚠️ 警告: configs/hsp_database.yaml 未找到。")
            return {"solvents": {}, "precursors": {}}
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def calc_hansen_distance(self, item1: dict, item2: dict) -> float:
        """核心计算：Hansen 空间距离 Ra"""
        ra_squared = 4 * (item1['dD'] - item2['dD'])**2 + \
                     (item1['dP'] - item2['dP'])**2 + \
                     (item1['dH'] - item2['dH'])**2
        return math.sqrt(ra_squared)

    def evaluate_solvent(self, precursor: str, solvent: str) -> Dict[str, Union[str, float]]:
        """评估单一前驱体在单一溶剂中的溶解能力"""
        if precursor not in self.db['precursors']:
            return {"error": f"前驱体 {precursor} 不在数据库中。"}
        if solvent not in self.db['solvents']:
            return {"error": f"溶剂 {solvent} 不在数据库中。"}

        p_data = self.db['precursors'][precursor]
        s_data = self.db['solvents'][solvent]
        
        ra = self.calc_hansen_distance(p_data, s_data)
        
        # 经验判断法则
        solubility = "Unknown"
        if ra <= 5.0:
            solubility = "Excellent (极易溶解)"
        elif ra <= 8.0:
            solubility = "Good (可溶解，可能需要加热或搅拌)"
        elif ra <= 12.0:
            solubility = "Poor (微溶，易析出沉淀)"
        else:
            solubility = "Insoluble (不溶，可作为完美的反溶剂候选)"

        return {
            "precursor": precursor,
            "solvent": solvent,
            "hansen_distance_Ra": round(ra, 2),
            "solubility_assessment": solubility
        }

    def recommend_solvent_system(self, precursor: str) -> Dict[str, Union[str, List[dict]]]:
        """
        高阶功能：为目标前驱体自动推荐最佳的【主溶剂】和【反溶剂】配方体系。
        """
        if precursor not in self.db['precursors']:
            return {"error": f"前驱体 {precursor} 不在数据库中。"}
            
        p_data = self.db['precursors'][precursor]
        hosts = []
        antis = []

        # 遍历所有溶剂，计算 Ra 并分类
        for s_name, s_data in self.db['solvents'].items():
            ra = self.calc_hansen_distance(p_data, s_data)
            entry = {"solvent": s_name, "Ra": round(ra, 2)}
            
            if s_data.get('type') == 'host':
                hosts.append(entry)
            elif s_data.get('type') == 'anti':
                antis.append(entry)

        # 主溶剂：Ra 越小越好 (升序排列)
        hosts.sort(key=lambda x: x['Ra'])
        # 反溶剂：Ra 越大越好，促使快速结晶 (降序排列)
        antis.sort(key=lambda x: x['Ra'], reverse=True)

        return {
            "target_precursor": precursor,
            "recommended_host_solvents": hosts[:3], # 推荐前三名主溶剂
            "recommended_antisolvents": antis[:3],  # 推荐前三名反溶剂
            "lab_advice": f"建议使用 {hosts[0]['solvent']} 溶解 {precursor} 制备前驱体墨水，并在旋涂的最后 10 秒滴加 {antis[0]['solvent']} 以诱导高质量薄膜结晶。"
        }

if __name__ == "__main__":
    expert = LabSynthesisExpert()
    
    # 1. 评估碘化铅在 DMF 中的溶解度
    print("--- 评估 PbI2 在 DMF 中的溶解度 ---")
    print(expert.evaluate_solvent("PbI2", "DMF"))

    # 2. 为难溶的 PbI2 推荐整套溶剂工艺体系
    print("\n--- PbI2 溶剂体系工艺推荐 ---")
    recommendation = expert.recommend_solvent_system("PbI2")
    import json
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))
