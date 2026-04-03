import os
import yaml
import warnings
from typing import Dict, Union

# 屏蔽 Pymatgen 警告
warnings.filterwarnings("ignore")
from pymatgen.core import Structure, Lattice

class CrystalGeneratorExpert:
    """
    晶体结构生成与材料发现专家。
    核心功能：
    1. 元素替换发现法 (Element Substitution)：基于已有的母体 CIF 结构，替换特定元素生成全新材料。
    2. 原型生成法 (Prototype Generation)：从零生成理想的无机双钙钛矿/单钙钛矿晶体晶胞。
    """
    def __init__(self):
        print("✨ [Discovery_Expert] Initializing Crystal Generator module...")
        self.config = self._load_config()
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 确保生成的虚拟材料有专门的存放点，或直接放入 cif_cache
        cache_rel_path = self.config.get('system', {}).get('cache_dir', './cache/cif_cache')
        self.save_dir = os.path.join(base_dir, cache_rel_path.replace("./", ""))
        os.makedirs(self.save_dir, exist_ok=True)

    def _load_config(self) -> dict:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, "configs", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def generate_by_substitution(self, base_cif_path: str, substitution_mapping: str) -> Dict[str, Union[str, bool]]:
        """
        基于元素替换策略生成新材料 CIF。
        :param base_cif_path: 母体材料的 CIF 路径
        :param substitution_mapping: 替换字典的字符串格式，如 "Pb:Sn, I:Br"
        """
        if not os.path.exists(base_cif_path):
            return {"error": f"找不到母体结构文件: {base_cif_path}"}

        try:
            # 1. 解析替换字典 (支持大模型传入的简单字符串格式)
            # 例如 "Pb:Sn, I:Br" -> {"Pb": "Sn", "I": "Br"}
            sub_dict = {}
            pairs = substitution_mapping.split(",")
            for pair in pairs:
                if ":" in pair:
                    k, v = pair.split(":")
                    sub_dict[k.strip()] = v.strip()
            
            if not sub_dict:
                return {"error": "解析替换规则失败，请确保格式如 'Pb:Sn, I:Br'"}

            # 2. 加载母体结构并执行替换
            structure = Structure.from_file(base_cif_path)
            structure.replace_species(sub_dict)
            
            # 获取新化学式
            new_formula = structure.composition.reduced_formula
            
            # 3. 保存新结构
            new_filename = f"AI_Generated_{new_formula}.cif"
            new_filepath = os.path.join(self.save_dir, new_filename)
            structure.to(filename=new_filepath, fmt="cif")

            return {
                "action": "Element Substitution",
                "new_formula": new_formula,
                "generated_cif_path": new_filepath,
                "note": "⚠️ 这是一个未经弛豫的初始结构（晶格常数继承自母体）。**极其重要**：在进行任何带隙或形成能计算前，必须先调用 CHGNet 工具对其进行几何优化 (task='relax')！"
            }
        except Exception as e:
            return {"error": f"生成新结构失败: {str(e)}"}

    def generate_cubic_perovskite(self, a_ion: str, b_ion: str, x_ion: str) -> Dict[str, Union[str, float]]:
        """
        从零生成理想的无机立方钙钛矿 (Pm-3m) 结构。
        :param a_ion: A位元素 (如 Cs, Rb)
        :param b_ion: B位元素 (如 Sn, Pb)
        :param x_ion: X位元素 (如 I, Br)
        """
        try:
            # 采用硬球模型近似估算初始晶格常数 a
            # 真实晶格常数需要 CHGNet 弛豫后才能准确获得，这里仅提供合理初值
            # 经验近似：a ≈ 2 * (R_B + R_X) (这里简化处理，假设 B-X 键长主导)
            from pymatgen.core.periodic_table import Element
            r_b = Element(b_ion).average_ionic_radius
            r_x = Element(x_ion).average_ionic_radius
            
            # 如果查不到半径，给一个典型初值 6.0 Angstrom
            a_lattice = 2.0 * (r_b + r_x) if (r_b and r_x) else 6.0
            
            # 创建 Pm-3m 立方晶胞
            lattice = Lattice.cubic(a_lattice)
            
            # 钙钛矿标准原子坐标 (分数坐标)
            species = [a_ion, b_ion, x_ion, x_ion, x_ion]
            coords = [
                [0.0, 0.0, 0.0],          # A site (角落)
                [0.5, 0.5, 0.5],          # B site (体心)
                [0.5, 0.5, 0.0],          # X site (面心)
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5]
            ]
            
            structure = Structure.from_spacegroup("Pm-3m", lattice, species, coords)
            new_formula = structure.composition.reduced_formula
            
            new_filename = f"AI_Generated_Cubic_{new_formula}.cif"
            new_filepath = os.path.join(self.save_dir, new_filename)
            structure.to(filename=new_filepath, fmt="cif")
            
            return {
                "action": "Prototype Generation",
                "spacegroup": "Pm-3m (Cubic)",
                "new_formula": new_formula,
                "initial_lattice_a_Angstrom": round(a_lattice, 2),
                "generated_cif_path": new_filepath,
                "note": "⚠️ 这是一个基于理想硬球模型的初始结构。必须先调用 CHGNet (task='relax') 进行结构弛豫，否则后续计算毫无意义！"
            }
        except Exception as e:
            return {"error": f"生成立方钙钛矿结构失败: {str(e)}"}

if __name__ == "__main__":
    expert = CrystalGeneratorExpert()
    
    # 场景 1：从零创造纯无机钙钛矿 CsSnI3
    print("--- 尝试生成新型立方相 CsSnI3 ---")
    res1 = expert.generate_cubic_perovskite("Cs", "Sn", "I")
    print(res1)
    
    # 场景 2：基于元素替换探索新材料
    # 假设我们有一个刚下载好的 CsSnI3 结构，我们想把 I 换成 Br 看看
    if "generated_cif_path" in res1:
        print("\n--- 尝试通过替换法生成 CsSnBr3 ---")
        res2 = expert.generate_by_substitution(res1["generated_cif_path"], "I:Br")
        print(res2)
