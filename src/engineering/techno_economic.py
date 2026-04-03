import os
import yaml
from typing import Dict, Union, List
from pymatgen.core import Composition, Element

class TechnoEconomicExpert:
    """
    技术经济与环保分析专家。
    功能：
    1. 解析化学配方，基于静态大宗数据库估算合成 1kg 材料的原料成本。
    2. 筛查配方中的剧毒元素 (如 Pb, Cd)，评估 RoHS 环保依从性。
    """
    def __init__(self):
        print("🏭 [TechnoEconomic_Expert] Initializing Cost & Toxicity Assessment module...")
        self.db = self._load_database()

    def _load_database(self) -> dict:
        """加载解耦的成本与毒性数据库"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        yaml_path = os.path.join(base_dir, "configs", "materials_cost_toxicity.yaml")
        
        if not os.path.exists(yaml_path):
            print("⚠️ 警告: configs/materials_cost_toxicity.yaml 未找到，使用内置默认估值。")
            return {"elements": {}, "compounds": {}}
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_item_data(self, item_name: str) -> dict:
        """从数据库中获取单项（元素或基团）的数据，若缺失则返回高昂的默认惩罚价"""
        if item_name in self.db.get('compounds', {}):
            return self.db['compounds'][item_name]
        if item_name in self.db.get('elements', {}):
            return self.db['elements'][item_name]
        
        # 默认未知元素的惩罚性估值，防止模型钻空子选用极冷门材料
        return {
            "price_usd_per_kg": 1000.0, 
            "is_toxic": False, 
            "rohs_restricted": False,
            "note": "数据库中无此元素报价，采用 1000 USD/kg 惩罚性估值"
        }

    def evaluate_material(self, formula: str) -> Dict[str, Union[float, str, bool, List[str]]]:
        """
        核心接口：同时评估化学式的原材料成本和毒性。
        支持类似 'MAPbI3' 或 'Cs2AgBiBr6' 的输入。
        """
        # 1. 预处理常见有机缩写，以便 Pymatgen 解析摩尔质量
        parseable_formula = formula
        # 注意：这里我们保留 MA 和 FA 作为独立单元去查价格库，
        # 但在计算摩尔质量(g/mol)时，必须把它们展开为具体原子以计算质量占比。
        mass_formula = formula 
        abbreviations = {"MA": "CH6N1", "FA": "CH5N2"}
        for abbr, full in abbreviations.items():
            mass_formula = mass_formula.replace(abbr, full)

        try:
            comp = Composition(mass_formula)
            total_mass = comp.weight # g/mol
        except Exception as e:
            return {"error": f"无法解析化学式 '{formula}'，请检查输入格式。({str(e)})"}

        # 2. 识别配方中包含的基础组件 (元素 + 有机基团)
        # 我们用正则表达式或简单的字符串匹配提取 MA/FA
        components = {}
        for abbr in abbreviations.keys():
            if abbr in parseable_formula:
                # 假设化学式中只包含 1 个该基团，如需精确匹配需更复杂的解析，这里简化处理
                components[abbr] = {"mass_g_mol": Composition(abbreviations[abbr]).weight, "amount": 1}
        
        # 提取无机元素
        for el, amt in comp.items():
            el_str = el.symbol
            # 过滤掉属于有机基团的 C, H, N，避免重复计价
            if el_str in ["C", "H", "N"] and components:
                continue
            components[el_str] = {"mass_g_mol": el.atomic_mass, "amount": amt}

        # 3. 汇总成本与毒性
        total_cost_per_kg = 0.0
        toxic_elements = []
        rohs_violations = []
        cost_breakdown = {}

        for item, data in components.items():
            db_data = self._get_item_data(item)
            
            # 计算该成分在 1kg 总材料中的质量占比
            mass_fraction = (data["mass_g_mol"] * data["amount"]) / total_mass
            
            # 计算该成分贡献的成本 (USD/kg)
            component_cost = mass_fraction * db_data["price_usd_per_kg"]
            total_cost_per_kg += component_cost
            
            cost_breakdown[item] = {
                "mass_fraction_percent": round(mass_fraction * 100, 2),
                "cost_contribution_usd": round(component_cost, 2)
            }

            # 毒性审查
            if db_data.get("is_toxic"):
                toxic_elements.append(item)
            if db_data.get("rohs_restricted"):
                rohs_violations.append(item)

        # 4. 生成商业化评估报告
        commercial_assessment = "商业化前景良好，成本低廉且无严重环保限制。"
        
        if total_cost_per_kg > 500:
            commercial_assessment = "成本过于高昂 (超过 500 USD/kg)，不适合大规模工业化量产，仅具有基础物理研究价值。"
        elif total_cost_per_kg > 100:
            commercial_assessment = "成本偏高，需评估其光电性能是否足以弥补高昂的制造成本。"
            
        if rohs_violations:
            commercial_assessment += f" ⚠️ 警告：含有 RoHS 严格限制的剧毒重金属 {rohs_violations}，存在极高的环保合规风险，可能无法在欧洲等地实现商业化。"
        elif toxic_elements:
            commercial_assessment += f" 注意：含有一定毒性的元素 {toxic_elements}，生产过程中需加强职业防护。"

        return {
            "target_formula": formula,
            "estimated_cost_usd_per_kg": round(total_cost_per_kg, 2),
            "cost_breakdown": cost_breakdown,
            "contains_toxic_elements": len(toxic_elements) > 0,
            "violates_RoHS_directive": len(rohs_violations) > 0,
            "commercial_assessment": commercial_assessment
        }

if __name__ == "__main__":
    expert = TechnoEconomicExpert()
    
    # 1. 测试经典的 MAPbI3 (极便宜但含铅)
    print("--- Testing MAPbI3 ---")
    print(expert.evaluate_material("MAPbI3"))

    # 2. 测试试图替换铅的 Cs2AgBiBr6 (无毒，但极其昂贵)
    print("\n--- Testing Cs2AgBiBr6 ---")
    print(expert.evaluate_material("Cs2AgBiBr6"))

    # 3. 测试 CIGS 铜铟镓硒 (经典的薄膜太阳能电池)
    print("\n--- Testing CuIn0.7Ga0.3Se2 ---")
    print(expert.evaluate_material("CuIn0.7Ga0.3Se2"))
