import os
import yaml
import numpy as np
import torch
import warnings

# 屏蔽 Pymatgen/CHGNet 的运行时警告
warnings.filterwarnings("ignore")

from pymatgen.core import Structure
from chgnet.model import CHGNet, StructOptimizer

# ==============================================================================
# 全局常量：Reference Energies (PBE Functional) - 用于计算形成能
# ==============================================================================
MP_REFERENCE_ENERGIES = {
    "H": -3.38, "Li": -1.91, "Be": -3.75, "B": -6.66, "C": -9.22, "N": -8.22, "O": -4.95, "F": -1.70,
    "Na": -1.31, "Mg": -1.53, "Al": -3.75, "Si": -5.42, "P": -5.41, "S": -4.13, "Cl": -1.82, "K": -1.13,
    "Ca": -1.97, "Ti": -7.72, "V": -8.99, "Cr": -9.51, "Mn": -9.04, "Fe": -8.31, "Co": -7.11, "Ni": -5.57,
    "Cu": -3.73, "Zn": -1.26, "Ga": -2.93, "Ge": -4.63, "As": -4.67, "Se": -3.50, "Br": -1.56,
    "Rb": -0.96, "Sr": -1.67, "Y": -6.50, "Zr": -8.54, "Nb": -10.23, "Mo": -10.95, "Tc": -10.42,
    "Ru": -9.28, "Rh": -7.36, "Pd": -5.23, "Ag": -2.83, "Cd": -0.92, "In": -2.66, "Sn": -3.99,
    "Sb": -4.13, "Te": -3.14, "I": -1.65, "Cs": -0.89, "Ba": -1.92, "Pb": -3.71, "Bi": -4.42
}

def load_config():
    """加载全局 YAML 配置文件获取物理计算参数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class CHGNetExpert:
    """
    物理底层模拟专家。
    基于预训练通用图神经网络力场 (CHGNet)，实现毫秒级的结构性质预测与几何弛豫。
    """
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚛️ [Physics_Expert] Initializing CHGNet on device: {self.device}...")
        
        self.config = load_config()
        self.fmax = self.config['physics'].get('chgnet_fmax', 0.05)
        self.steps = self.config['physics'].get('chgnet_steps', 500)
        
        try:
            self.model = CHGNet.load()
            print("✅ [Physics_Expert] CHGNet Model loaded successfully.")
        except Exception as e:
            print(f"❌ [Physics_Expert] Failed to load model: {e}")
            raise e

    def _load_structure(self, cif_path: str) -> Structure:
        """从 CIF 文件路径加载 Pymatgen Structure 对象"""
        if not os.path.exists(cif_path):
            raise FileNotFoundError(f"CIF文件不存在: {cif_path}")
        return Structure.from_file(cif_path)

    def _calc_formation_energy(self, total_energy_ev: float, composition) -> dict:
        """内部逻辑：基于 MP 元素参考能量计算形成能"""
        total_atoms = composition.num_atoms
        ref_sum = 0.0
        missing = []
        
        for el, amt in composition.items():
            el_str = str(el)
            if el_str in MP_REFERENCE_ENERGIES:
                ref_sum += MP_REFERENCE_ENERGIES[el_str] * amt
            else:
                missing.append(el_str)
        
        if missing:
            return {"formation_energy_eV_per_atom": None, "warning": f"缺少参考能量: {missing}"}
        
        e_form = (total_energy_ev - ref_sum) / total_atoms
        return {
            "formation_energy_eV_per_atom": round(float(e_form), 4),
            "is_thermodynamically_stable": e_form < 0
        }

    def predict_properties(self, cif_path: str, task_list: list = None) -> dict:
        """
        核心计算接口：按需预测材料物理性质 (形成能、磁性、应力、受力)。
        供 Agent 作为工具调用。
        """
        try:
            structure = self._load_structure(cif_path)
            prediction = self.model.predict_structure(structure)
            total_energy = float(prediction['e'])
            
            # 基础信息
            result = {
                "composition": structure.composition.formula,
                "total_energy_eV": round(total_energy, 4),
            }

            if not task_list:
                task_list = ["formation"] # 默认只算形成能

            # 1. 形成能 (Formation Energy / Stability)
            if any(k in task_list for k in ['form', 'formation', 'stability']):
                result.update(self._calc_formation_energy(total_energy, structure.composition))

            # 2. 磁矩 (Magnetic Moments)
            if any(k in task_list for k in ['mag', 'magnetic']):
                m = prediction.get('m')
                if m is not None:
                    result["avg_magnetic_moment_muB"] = round(float(np.mean(np.abs(m))), 4)
                    result["total_magnetization_muB"] = round(float(np.sum(m)), 4)
                else:
                    result["magnetic_info"] = "Non-magnetic or not predicted."

            # 3. 应力 (Stress)
            if any(k in task_list for k in ['stress', 'pressure']):
                s = prediction['s'] * 0.1 # kBar -> GPa
                result["max_stress_GPa"] = round(float(np.max(np.abs(s))), 4)

            # 4. 原子受力 (Forces)
            if any(k in task_list for k in ['force', 'relax']):
                f = prediction['f']
                result["max_atomic_force_eV_A"] = round(float(np.max(np.abs(f))), 4)

            return result
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def optimize_structure(self, cif_path: str) -> dict:
        """
        结构弛豫接口：优化晶格和原子位置至能量最低状态。
        """
        try:
            structure = self._load_structure(cif_path)
            relaxer = StructOptimizer(model=self.model, optimizer_class="BFGS")
            
            print(f"⚙️ [Physics_Expert] Relaxing {structure.composition.formula}...")
            result = relaxer.relax(structure, fmax=self.fmax, steps=self.steps)
            
            final_struct = result['final_structure']
            final_energy = float(result['trajectory'].energies[-1])
            
            # 覆盖原文件或保存为新文件均可，这里选择返回状态
            return {
                "status": "success",
                "initial_energy_eV": round(float(result['trajectory'].energies[0]), 4),
                "final_energy_eV": round(final_energy, 4),
                "energy_change_eV": round(final_energy - float(result['trajectory'].energies[0]), 4)
            }
        except Exception as e:
            return {"error": f"Optimization failed: {str(e)}"}

if __name__ == "__main__":
    # 单元测试 (创建一个虚拟的盐岩结构测试 CHGNet 是否可用)
    try:
        from pymatgen.core import Lattice
        # 在当前目录生成一个临时测试文件
        test_cif = "test_NaCl.cif"
        dummy = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.6), ["Na", "Cl"], [[0,0,0], [0.5,0.5,0.5]])
        dummy.to(filename=test_cif, fmt="cif")
        
        expert = CHGNetExpert()
        
        print("\n--- Test 1: Prediction ---")
        res = expert.predict_properties(test_cif, task_list=["formation", "mag"])
        print(res)
        
        print("\n--- Test 2: Relaxation ---")
        res_opt = expert.optimize_structure(test_cif)
        print(res_opt)
        
        # 清理临时文件
        if os.path.exists(test_cif):
            os.remove(test_cif)
            
    except Exception as e:
        print(f"测试失败: {e}")
