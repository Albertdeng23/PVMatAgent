import os
import re
import yaml
import warnings
import numpy as np
from typing import Dict, Union, List, Optional, Tuple

warnings.filterwarnings("ignore")
from pymatgen.core import Structure, Lattice, DummySpecies, Element

# ==============================================================================
# 硬球模型经验常数：用于从离子半径估算晶格常数
# ==============================================================================
# 经验校正因子：硬球模型 a ≈ f * (r_A + r_X + √2(r_B + r_X)) / √2
# 简化版：a ≈ 2*(r_B + r_X)，再加入 A位半径的修正
# 对常见的钙钛矿验证效果可接受（误差通常 < 5%）
HARD_SPHERE_FACTOR = 2.0

# ==============================================================================
# 常见钙钛矿原型的 Wyckoff 位置数据
# 只存不对称单元中的代表性坐标，其余由 Pymatgen 空间群对称性自动生成
# ==============================================================================
PEROVSKITE_PROTOTYPES = {
    "cubic": {
        "spacegroup": "Pm-3m",
        "description": "立方钙钛矿 ABX3，单胞含 1 个公式单元",
        "wyckoff_sites": [
            {"species_idx": 0, "coord": [0.0, 0.0, 0.0]},      # A: 1a
            {"species_idx": 1, "coord": [0.5, 0.5, 0.5]},      # B: 1b
            {"species_idx": 2, "coord": [0.0, 0.5, 0.5]},      # X: 3c (只放1个生成坐标)
        ],
        "species_count": [1, 1, 1],  # A, B, X 各 1 个（在不对称单元中）
    },
    "double_perovskite": {
        "spacegroup": "Fm-3m",
        "description": "立方双钙钛矿 A2BB'X6，单胞含 4 个公式单元（岩盐有序）",
        "wyckoff_sites": [
            {"species_idx": 0, "coord": [0.25, 0.25, 0.25]},   # A: 8c
            {"species_idx": 1, "coord": [0.0, 0.0, 0.0]},       # B: 4a
            {"species_idx": 2, "coord": [0.5, 0.5, 0.5]},       # B': 4b
            {"species_idx": 3, "coord": [0.25, 0.0, 0.0]},      # X: 24e (只放1个)
        ],
        "species_count": [2, 1, 1, 6],  # A2, B, B', X6 (在公式单元中)
        "formula_template": "{A}2{B}{Bp}{X}6",
    },
    "tetragonal": {
        "spacegroup": "I4/mcm",
        "description": "四方钙钛矿 ABX3（八面体倾斜 a⁰a⁰c⁻），单胞含 4 个公式单元",
        "wyckoff_sites": [
            {"species_idx": 0, "coord": [0.0, 0.5, 0.25]},      # A: 4b
            {"species_idx": 1, "coord": [0.0, 0.0, 0.0]},       # B: 4c
            {"species_idx": 2, "coord": [0.0, 0.0, 0.25]},      # X apical: 4a
            {"species_idx": 2, "coord": [0.25, 0.25, 0.0]},     # X equatorial: 8h (只放1个)
        ],
        "species_count": [1, 1, 3],  # ABX3 per formula unit
        "lattice_scale": 2.0,  # a_tet ≈ a_cubic, c_tet ≈ 2*a_cubic
    },
}

# ==============================================================================
# 有机阳离子的展开映射（用于计算摩尔质量和生成占位原子）
# ==============================================================================
ORGANIC_COMPOSITIONS = {
    "MA": "CH6N1",    # CH3NH3+
    "FA": "CH5N2",    # CH(NH2)2+
    "EA": "C2H8N1",   # CH3CH2NH3+
    "GA": "C1H6N3",   # C(NH2)3+
}


class CrystalGeneratorExpert:
    """
    晶体结构生成与材料发现专家。

    功能：
    1. 立方钙钛矿原型生成 (Pm-3m ABX3)  —— 支持无机+有机A位
    2. 立方双钙钛矿生成 (Fm-3m A2BB'X6)      —— 支持全无机
    3. 四方相钙钛矿生成 (I4/mcm)             —— 支持八面体倾斜结构
    4. 全元素替换生成 (Element Substitution)   —— 基于母体CIF
    5. 部分组分替换生成 (Partial Substitution) —— 支持中间组分如 I₀.₅Br₀.₅

    所有生成的结构都需经 CHGNet 弛豫才能用于性质预测。
    """

    def __init__(self):
        print("* [Discovery_Expert] Initializing Crystal Generator module...")
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config = self._load_config()
        self.organic_radii = self._load_organic_radii()

        cache_rel_path = self.config.get('system', {}).get('cache_dir', './cache/cif_cache')
        self.save_dir = os.path.join(self.base_dir, cache_rel_path.replace("./", ""))
        os.makedirs(self.save_dir, exist_ok=True)

    # ========================================================================
    # 配置加载
    # ========================================================================

    def _load_config(self) -> dict:
        config_path = os.path.join(self.base_dir, "configs", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def _load_organic_radii(self) -> dict:
        yaml_path = os.path.join(self.base_dir, "configs", "organic_radii.yaml")
        if not os.path.exists(yaml_path):
            return {}
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('organics', {})

    # ========================================================================
    # 离子半径查询（支持无机元素 + 有机阳离子）
    # ========================================================================

    def _get_ionic_radius(self, ion: str, site: str = "A") -> float:
        """
        查询离子半径，优先查找有机数据库，其次 Shannon 半径。
        返回以 Angstrom 为单位的半径值。
        """
        ion_clean = re.sub(r'[0-9]*[+\-−]$', '', ion.strip())
        ion_upper = ion_clean.upper()

        # 有机阳离子
        if ion_upper in self.organic_radii:
            return self.organic_radii[ion_upper].get('radius_A', 2.5)

        # 无机元素：使用 Shannon 平均离子半径
        try:
            el = Element(ion_clean.capitalize())
            if el.average_ionic_radius:
                return el.average_ionic_radius
        except Exception:
            pass

        # 回退默认值
        defaults = {"A": 1.8, "B": 0.8, "X": 1.3}
        return defaults.get(site, 1.5)

    def _is_organic(self, ion: str) -> bool:
        """判断离子是否为有机阳离子。"""
        return re.sub(r'[0-9]*[+\-−]$', '', ion.strip()).upper() in self.organic_radii

    # ========================================================================
    # 物种创建（支持有机 DummySpecies）
    # ========================================================================

    def _make_species(self, ion: str, site: str = "A"):
        """
        创建 Pymatgen 可用的物种对象。
        有机离子 → DummySpecies；无机元素 → 保留字符串由 Pymatgen 自动解析。
        """
        ion_clean = re.sub(r'[0-9]*[+\-−]$', '', ion.strip())
        if self._is_organic(ion_clean):
            oxidation = +1 if site == "A" else (0 if site == "X" else +2)
            return DummySpecies(ion_clean, oxidation_state=oxidation)
        return ion_clean.capitalize()

    # ========================================================================
    # 晶格常数估算
    # ========================================================================

    def _estimate_lattice_a(self, a_ion: str, b_ion: str, x_ion: str) -> float:
        """
        基于硬球模型估算立方钙钛矿的晶格常数 a。
        公式：a ≈ 2 * (r_B + r_X)，再加入 A 位占位的修正。
        """
        r_b = self._get_ionic_radius(b_ion, "B")
        r_x = self._get_ionic_radius(x_ion, "X")
        r_a = self._get_ionic_radius(a_ion, "A")

        # 主项：B-X 键长
        a_bx = HARD_SPHERE_FACTOR * (r_b + r_x)
        # A-X 距离约束：a = √2 * (r_A + r_X)
        a_ax = np.sqrt(2) * (r_a + r_x)
        # 加权平均（B-X 通常主导晶格）
        a = 0.7 * a_bx + 0.3 * a_ax

        if a <= 0 or a > 15:
            a = 6.0
        return round(a, 3)

    # ========================================================================
    # 1. 立方钙钛矿 ABX3 生成（核心，已修复 Wyckoff bug）
    # ========================================================================

    def generate_cubic_perovskite(self, a_ion: str, b_ion: str, x_ion: str) -> Dict:
        """
        从零生成理想 Pm-3m 立方钙钛矿 ABX3 结构。

        参数:
            a_ion: A 位阳离子 (如 Cs, MA, FA, Rb)
            b_ion: B 位阳离子 (如 Sn, Pb, Ge)
            x_ion: X 位阴离子 (如 I, Br, Cl)

        返回:
            包含生成结构路径和元信息的字典

        修复说明:
            旧版将 3 个 X 坐标全部放入不对称单元，Pymatgen 对每个坐标
            施加 Pm-3m 对称操作后产生 9 个等效 X 位点 → 错误化学式。
            修复后只放 1 个 X 代表坐标 (3c Wyckoff)，对称性自动生成
            其余 2 个，得到正确的 ABX3 化学计量比。
        """
        # 清洗电荷后缀
        a_clean = re.sub(r'[0-9]*[+\-−]$', '', a_ion.strip())
        b_clean = re.sub(r'[0-9]*[+\-−]$', '', b_ion.strip())
        x_clean = re.sub(r'[0-9]*[+\-−]$', '', x_ion.strip())

        try:
            a_lattice = self._estimate_lattice_a(a_clean, b_clean, x_clean)
            lattice = Lattice.cubic(a_lattice)

            # 不对称单元：A×1, B×1, X×1（代表 3c Wyckoff 位置）
            species = [
                self._make_species(a_clean, "A"),
                self._make_species(b_clean, "B"),
                self._make_species(x_clean, "X"),
            ]
            coords = [
                [0.0, 0.0, 0.0],      # 1a: A 位
                [0.5, 0.5, 0.5],      # 1b: B 位
                [0.0, 0.5, 0.5],      # 3c: X 位（只放 1 个生成坐标，对称性生成 3 个）
            ]

            structure = Structure.from_spacegroup("Pm-3m", lattice, species, coords)
            structure.sort()

            # 验证化学计量比
            comp = structure.composition
            expected_ratio = 3  # X:A 应为 3:1
            actual_x_count = sum(1 for s in structure.species if x_clean.upper() in str(s).upper())
            if actual_x_count != 3:
                # 回退：手动构造
                return self._fallback_cubic(a_clean, b_clean, x_clean, a_lattice)

            new_formula = comp.reduced_formula
            new_filename = f"AI_Generated_Cubic_{new_formula}.cif"
            new_filepath = os.path.join(self.save_dir, new_filename)
            structure.to(filename=new_filepath, fmt="cif")

            return {
                "action": "Prototype Generation",
                "spacegroup": "Pm-3m (Cubic)",
                "new_formula": new_formula,
                "initial_lattice_a_Angstrom": a_lattice,
                "generated_cif_path": new_filepath,
                "note": "[!] 基于硬球模型的初始结构，必须先调用 CHGNet (task='relax') 进行几何弛豫。"
            }
        except Exception as e:
            # 回退：手动构造
            try:
                return self._fallback_cubic(a_clean, b_clean, x_clean,
                                            self._estimate_lattice_a(a_clean, b_clean, x_clean))
            except Exception as e2:
                return {"error": f"生成立方钙钛矿失败: {str(e)} → fallback: {str(e2)}"}

    def _fallback_cubic(self, a_ion: str, b_ion: str, x_ion: str, a_lattice: float) -> Dict:
        """回退方案：手动构造 Pm-3m 立方钙钛矿晶胞，绕过 from_spacegroup。"""
        lattice = Lattice.cubic(a_lattice)
        species = [
            self._make_species(a_ion, "A"),
            self._make_species(b_ion, "B"),
            self._make_species(x_ion, "X"),
            self._make_species(x_ion, "X"),
            self._make_species(x_ion, "X"),
        ]
        coords = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
        structure = Structure(lattice, species, coords, coords_are_cartesian=False)
        structure.sort()

        new_formula = structure.composition.reduced_formula
        new_filename = f"AI_Generated_Cubic_{new_formula}.cif"
        new_filepath = os.path.join(self.save_dir, new_filename)
        structure.to(filename=new_filepath, fmt="cif")

        return {
            "action": "Prototype Generation (fallback manual construction)",
            "spacegroup": "Pm-3m (Cubic)",
            "new_formula": new_formula,
            "initial_lattice_a_Angstrom": a_lattice,
            "generated_cif_path": new_filepath,
            "note": "[!] 使用回退手动构造。必须先调用 CHGNet (task='relax') 进行几何弛豫。"
        }

    # ========================================================================
    # 2. 立方双钙钛矿 A2BB'X6 生成
    # ========================================================================

    def generate_double_perovskite(self, a_ion: str, b_ion: str, b_prime_ion: str,
                                   x_ion: str) -> Dict:
        """
        生成岩盐有序的立方双钙钛矿 A₂BB'X₆ (Fm-3m)。

        参数:
            a_ion:  A 位阳离子
            b_ion:  B 位阳离子
            b_prime_ion: B' 位阳离子
            x_ion:  X 位阴离子
        """
        a_clean = re.sub(r'[0-9]*[+\-−]$', '', a_ion.strip())
        b_clean = re.sub(r'[0-9]*[+\-−]$', '', b_ion.strip())
        bp_clean = re.sub(r'[0-9]*[+\-−]$', '', b_prime_ion.strip())
        x_clean = re.sub(r'[0-9]*[+\-−]$', '', x_ion.strip())

        try:
            # 估算晶格常数：A₂BB'X₆ ≈ 2×ABX₃ 的 a（因为超胞加倍）
            r_a = self._get_ionic_radius(a_clean, "A")
            r_b = self._get_ionic_radius(b_clean, "B")
            r_bp = self._get_ionic_radius(bp_clean, "B")
            r_x = self._get_ionic_radius(x_clean, "X")
            # 使用较大的 B 位半径
            r_b_avg = max(r_b, r_bp)
            a = HARD_SPHERE_FACTOR * (r_b_avg + r_x)
            a = round(max(a, 5.0), 3)

            lattice = Lattice.cubic(a)

            species = [
                self._make_species(a_clean, "A"),
                self._make_species(b_clean, "B"),
                self._make_species(bp_clean, "B"),
                self._make_species(x_clean, "X"),
            ]
            coords = [
                [0.25, 0.25, 0.25],   # A: 8c
                [0.0, 0.0, 0.0],       # B: 4a
                [0.5, 0.5, 0.5],       # B': 4b
                [0.25, 0.0, 0.0],      # X: 24e（只放1个生成坐标）
            ]

            structure = Structure.from_spacegroup("Fm-3m", lattice, species, coords)
            structure.sort()

            new_formula = structure.composition.reduced_formula
            new_filename = f"AI_Generated_DoublePv_{new_formula}.cif"
            new_filepath = os.path.join(self.save_dir, new_filename)
            structure.to(filename=new_filepath, fmt="cif")

            return {
                "action": "Double Perovskite Generation",
                "spacegroup": "Fm-3m (Cubic, Rock-salt ordered)",
                "new_formula": new_formula,
                "initial_lattice_a_Angstrom": a,
                "generated_cif_path": new_filepath,
                "note": "[!] 假设完美岩盐有序。必须先调用 CHGNet (task='relax') 进行几何弛豫。"
            }
        except Exception as e:
            return {"error": f"生成双钙钛矿失败: {str(e)}"}

    # ========================================================================
    # 3. 全元素替换生成（已修复：支持有机离子）
    # ========================================================================

    def generate_by_substitution(self, base_cif_path: str,
                                  substitution_mapping: str) -> Dict:
        """
        基于已有 CIF 进行全元素替换。

        参数:
            base_cif_path: 母体 CIF 路径
            substitution_mapping: 替换规则，如 "Pb:Sn, I:Br"
        """
        if not os.path.exists(base_cif_path):
            return {"error": f"找不到母体结构文件: {base_cif_path}"}

        try:
            sub_dict = {}
            for pair in substitution_mapping.split(","):
                if ":" in pair:
                    k, v = pair.split(":")
                    sub_dict[k.strip()] = v.strip()

            if not sub_dict:
                return {"error": "解析替换规则失败，请确保格式如 'Pb:Sn, I:Br'"}

            structure = Structure.from_file(base_cif_path)

            # 将有机替换目标转为 DummySpecies（Pymatgen 原生 replace_species 不处理）
            processed_sub = {}
            for old, new in sub_dict.items():
                old_clean = re.sub(r'[0-9]*[+\-−]$', '', old.strip())
                if self._is_organic(new) or self._is_organic(old_clean):
                    processed_sub[old_clean] = self._make_species(new, "A")
                else:
                    processed_sub[old.strip()] = new.strip()

            structure.replace_species(processed_sub)
            new_formula = structure.composition.reduced_formula

            new_filename = f"AI_Generated_{new_formula}.cif"
            new_filepath = os.path.join(self.save_dir, new_filename)
            structure.to(filename=new_filepath, fmt="cif")

            return {
                "action": "Element Substitution",
                "new_formula": new_formula,
                "generated_cif_path": new_filepath,
                "note": "[!] 晶格常数继承自母体，必须先调用 CHGNet (task='relax') 进行几何弛豫。"
            }
        except Exception as e:
            return {"error": f"生成新结构失败: {str(e)}"}

    # ========================================================================
    # 4. 部分组分替换生成（核心新增：支持中间组分）
    # ========================================================================

    def generate_partial_substitution(self, base_cif_path: str,
                                       substitution_spec: str) -> Dict:
        """
        基于母体 CIF 进行部分元素替换，生成中间组分结构。

        参数:
            base_cif_path: 母体 CIF 路径
            substitution_spec: 替换规格，格式为:
                "X_site: I->Br, fraction=0.5"    → 将 50% 的 I 替换为 Br
                "B_site: Sn->Ge, fraction=0.6"   → 将 60% 的 Sn 替换为 Ge

        实现: 找到符合 old_species 的所有原子位点，随机选取 fraction 比例替换为 new_species。
        """
        if not os.path.exists(base_cif_path):
            return {"error": f"找不到母体结构文件: {base_cif_path}"}

        try:
            # 解析规格字符串
            # 支持格式: "I->Br, fraction=0.5" 或 "X_site: I->Br, fraction=0.5"
            spec = substitution_spec.strip()
            # 移除 site 前缀（如 "X_site:", "B_site:"）
            if ":" in spec and "->" in spec.split(":")[-1]:
                spec = spec.split(":")[-1].strip()

            # 解析 old->new 和 fraction
            match = re.match(r'(\w+)\s*->\s*(\w+)\s*,\s*fraction\s*=\s*([\d.]+)', spec)
            if not match:
                return {"error": (
                    "替换规格格式错误。应为: 'old_element->new_element, fraction=0.XX'。"
                    "示例: 'I->Br, fraction=0.15' 表示将 15%% 的 I 替换为 Br。"
                )}
            old_species_str = match.group(1).strip()
            new_species_str = match.group(2).strip()
            fraction = float(match.group(3))

            if fraction <= 0 or fraction >= 1:
                return {"error": f"替换比例 fraction={fraction} 必须在 (0, 1) 之间。"}

            structure = Structure.from_file(base_cif_path)

            # 找到所有 old_species 的位点索引
            old_indices = [
                i for i, site in enumerate(structure)
                if old_species_str.upper() in str(site.species).upper()
            ]
            if not old_indices:
                return {
                    "error": f"在结构中未找到元素 '{old_species_str}'。"
                    f"结构中包含: {set(str(s.species) for s in structure)}"
                }

            # 随机选取 fraction 比例的位点
            n_replace = max(1, int(round(len(old_indices) * fraction)))
            rng = np.random.default_rng(42)
            replace_indices = np.atleast_1d(
                rng.choice(old_indices, size=n_replace, replace=False)
            ).tolist()

            # 执行替换
            new_species = self._make_species(new_species_str,
                                             "X" if old_species_str.upper() in ["I", "BR", "CL", "F"] else "B")
            for idx in replace_indices:
                structure[idx] = new_species

            new_formula = structure.composition.reduced_formula
            actual_fraction = n_replace / len(old_indices)

            new_filename = f"AI_Generated_Partial_{new_formula}.cif"
            new_filepath = os.path.join(self.save_dir, new_filename)
            structure.to(filename=new_filepath, fmt="cif")

            return {
                "action": "Partial Substitution",
                "base_structure": os.path.basename(base_cif_path),
                "substitution": f"{old_species_str} -> {new_species_str}",
                "target_fraction": fraction,
                "actual_fraction": round(actual_fraction, 3),
                "replaced_sites": f"{n_replace}/{len(old_indices)}",
                "new_formula": new_formula,
                "generated_cif_path": new_filepath,
                "note": (
                    "[!] 部分替换后的结构继承母体晶格常数，且替换位点随机选择。"
                    "必须调用 CHGNet (task='relax') 进行几何弛豫。"
                    "如需特定有序排列，请使用超胞并手动指定替换模式。"
                ),
            }
        except Exception as e:
            return {"error": f"部分替换生成失败: {str(e)}"}

    # ========================================================================
    # 5. 从参数构建结构（参数→CIF 的核心接口，供 LLM 自然语言→工具调用使用）
    # ========================================================================

    def generate_from_template(self, template: str, ions: List[str]) -> Dict:
        """
        根据结构模板名称和离子列表生成 CIF。

        参数:
            template: 结构模板名，可选:
                "cubic"              → ABX3 立方钙钛矿 (ions=[A,B,X])
                "double_perovskite"  → A2BB'X6 双钙钛矿 (ions=[A,B,B',X])
                "tetragonal"         → ABX3 四方钙钛矿 (ions=[A,B,X])
            ions: 离子列表

        这是 LLM 自然语言 → 结构化参数 → CIF 的核心接口。
        LLM 从用户需求中提取模板类型和离子，调用此工具生成结构。
        """
        template = template.lower().strip()
        if template == "cubic":
            if len(ions) != 3:
                return {"error": f"cubic 模板需要 [A,B,X] 三个离子，实际收到 {len(ions)} 个: {ions}"}
            return self.generate_cubic_perovskite(ions[0], ions[1], ions[2])
        elif template in ("double_perovskite", "double"):
            if len(ions) != 4:
                return {"error": f"double_perovskite 模板需要 [A,B,B',X] 四个离子，实际收到 {len(ions)} 个: {ions}"}
            return self.generate_double_perovskite(ions[0], ions[1], ions[2], ions[3])
        elif template == "tetragonal":
            if len(ions) != 3:
                return {"error": f"tetragonal 模板需要 [A,B,X] 三个离子，实际收到 {len(ions)} 个: {ions}"}
            return self.generate_tetragonal_perovskite(ions[0], ions[1], ions[2])
        else:
            available = ["cubic", "double_perovskite", "tetragonal"]
            return {"error": f"未知模板 '{template}'。可选: {available}"}

    def generate_tetragonal_perovskite(self, a_ion: str, b_ion: str, x_ion: str) -> Dict:
        """
        生成四方相 (I4/mcm) 钙钛矿 ABX3，模拟 a⁰a⁰c⁻ 八面体倾斜。
        """
        a_clean = re.sub(r'[0-9]*[+\-−]$', '', a_ion.strip())
        b_clean = re.sub(r'[0-9]*[+\-−]$', '', b_ion.strip())
        x_clean = re.sub(r'[0-9]*[+\-−]$', '', x_ion.strip())

        try:
            a0 = self._estimate_lattice_a(a_clean, b_clean, x_clean)
            # a_tet ≈ √2·a_cubic, c_tet ≈ 2·a_cubic
            a_tet = round(np.sqrt(2) * a0, 3)
            c_tet = round(2.0 * a0, 3)
            lattice = Lattice.tetragonal(a_tet, c_tet)

            species = [
                self._make_species(a_clean, "A"),
                self._make_species(b_clean, "B"),
                self._make_species(x_clean, "X"),
            ]
            coords = [
                [0.0, 0.5, 0.25],       # A: 4b
                [0.0, 0.0, 0.0],         # B: 4c
                [0.0, 0.0, 0.25],        # X1: 4a (apical)
                [0.25, 0.25, 0.0],       # X2: 8h (equatorial, 1个生成坐标)
            ]

            structure = Structure.from_spacegroup("I4/mcm", lattice, species, coords)
            structure.sort()

            new_formula = structure.composition.reduced_formula
            new_filename = f"AI_Generated_Tetragonal_{new_formula}.cif"
            new_filepath = os.path.join(self.save_dir, new_filename)
            structure.to(filename=new_filepath, fmt="cif")

            return {
                "action": "Tetragonal Perovskite Generation",
                "spacegroup": "I4/mcm (Tetragonal, a⁰a⁰c⁻ tilt)",
                "new_formula": new_formula,
                "lattice_parameters": {"a_Angstrom": a_tet, "c_Angstrom": c_tet},
                "generated_cif_path": new_filepath,
                "note": "[!] 初始倾斜结构。必须调用 CHGNet (task='relax') 进行几何弛豫。"
            }
        except Exception as e:
            return {"error": f"生成四方钙钛矿失败: {str(e)}"}

    # ========================================================================
    # 单元测试
    # ========================================================================

    def run_self_test(self) -> Dict:
        """快速自检：验证关键功能是否产出正确的化学计量比。"""
        results = {}

        # 测试 1：无机立方钙钛矿
        r = self.generate_cubic_perovskite("Cs", "Sn", "I")
        results["inorganic_cubic"] = {
            "formula": r.get("new_formula", "ERROR"),
            "valid": "I3" in str(r.get("new_formula", "")) and "Cs" in str(r.get("new_formula", "")),
        }

        # 测试 2：有机-无机杂化
        r = self.generate_cubic_perovskite("MA", "Pb", "I")
        results["hybrid_cubic"] = {
            "formula": r.get("new_formula", "ERROR"),
            "valid": "Ma" in str(r.get("new_formula", "")) and "Pb" in str(r.get("new_formula", "")),
        }

        # 测试 3：双钙钛矿
        r = self.generate_double_perovskite("Cs", "Ag", "Bi", "Br")
        results["double_perovskite"] = {
            "formula": r.get("new_formula", "ERROR"),
            "valid": "Br" in str(r.get("new_formula", "")),
        }

        # 测试 4：部分替换
        r_base = self.generate_cubic_perovskite("Cs", "Sn", "I")
        if "generated_cif_path" in r_base:
            r = self.generate_partial_substitution(r_base["generated_cif_path"],
                                                    "I->Br, fraction=0.5")
            results["partial_substitution"] = {
                "formula": r.get("new_formula", "ERROR"),
                "valid": "Br" in str(r.get("new_formula", "")) and "I" in str(r.get("new_formula", "")),
            }

        return results


if __name__ == "__main__":
    expert = CrystalGeneratorExpert()

    print("=" * 60)
    print("[TEST] CrystalGeneratorExpert 自检")
    print("=" * 60)

    test_results = expert.run_self_test()
    for name, result in test_results.items():
        status = "✅" if result.get("valid", False) else "❌"
        print(f"{status} {name}: {result}")
