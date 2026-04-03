import numpy as np
import warnings
from scipy.integrate import simpson
from scipy.optimize import fsolve
from scipy.constants import h, c, k, e

warnings.filterwarnings("ignore")

class SLMEExpert:
    """
    高阶光伏物理专家：SLME (光谱限制最大效率) 计算器。
    相比于理想的 S-Q 极限，SLME 考虑了：
    1. 材料的真实光学吸收率 (基于薄膜厚度 L 和 Tauc 吸收模型)。
    2. 间接带隙材料的非辐射复合惩罚 (基于直接带隙与最小带隙的能量差)。
    """
    def __init__(self, T_cell=300, T_sun=5778):
        self.T_cell = T_cell
        self.T_sun = T_sun
        self.omega_sun = 6.85e-5 # 太阳固体角
        print("🔭 [Advanced_PV_Expert] Initializing SLME (Spectroscopic Limited Maximum Efficiency) module...")

    def _planck_photon_flux(self, energy_ev, temp):
        """计算普朗克黑体辐射光子通量密度 (photons / s*m^2*eV)"""
        energy_j = energy_ev * e
        # 防止溢出
        exponent = np.clip(energy_j / (k * temp), a_min=None, a_max=700)
        denominator = np.exp(exponent) - 1
        flux = (2 * np.pi * (energy_j**2)) / ((h**3) * (c**2) * denominator)
        return flux * e

    def _simulate_absorption_spectrum(self, energies_ev, Eg, Eg_direct, A_direct=1e5, A_indirect=1e4):
        """
        利用 Tauc 模型动态模拟吸收光谱 \alpha(E) (单位: cm^-1)
        - 直接带隙跃迁: \alpha(E) \propto (E - Eg_direct)^(1/2) / E
        - 间接带隙跃迁: \alpha(E) \propto (E - Eg_indirect)^2 / E
        """
        alpha = np.zeros_like(energies_ev)
        
        # 1. 如果有间接带隙部分 (Eg < Eg_direct)
        if Eg < Eg_direct:
            mask_ind = (energies_ev >= Eg) & (energies_ev < Eg_direct)
            if np.any(mask_ind):
                E_ind = energies_ev[mask_ind]
                alpha[mask_ind] = A_indirect * ((E_ind - Eg)**2) / E_ind
                
        # 2. 直接带隙吸收部分 (主要吸收贡献)
        mask_dir = energies_ev >= Eg_direct
        if np.any(mask_dir):
            E_dir = energies_ev[mask_dir]
            # 叠加间接吸收的尾巴和直接吸收的主体
            alpha_ind_part = A_indirect * ((E_dir - Eg)**2) / E_dir if Eg < Eg_direct else 0
            alpha_dir_part = A_direct * np.sqrt(E_dir - Eg_direct) / E_dir
            alpha[mask_dir] = alpha_dir_part + alpha_ind_part
            
        return alpha

    def calculate_slme(self, bandgap_ev: float, direct_bandgap_ev: float, thickness_um: float = 2.0):
        """
        核心计算：SLME 理论效率。
        :param bandgap_ev: 材料的最小带隙 (Eg)
        :param direct_bandgap_ev: 材料的最小直接允许跃迁带隙 (Eg^da)
        :param thickness_um: 光伏吸收层的薄膜厚度 (微米, um)
        """
        if bandgap_ev <= 0 or direct_bandgap_ev <= 0:
            return {"error": "带隙必须大于 0 eV"}
        if direct_bandgap_ev < bandgap_ev:
            direct_bandgap_ev = bandgap_ev # 物理修正：直接带隙不可能小于基础带隙

        # 1. 建立能量网格
        energies = np.linspace(0.1, 5.0, 2000)
        
        # 2. 获取吸收系数 \alpha(E) 并计算吸收率 a(E)
        # 将厚度转为 cm (1 um = 1e-4 cm)
        L_cm = thickness_um * 1e-4 
        alpha_E = self._simulate_absorption_spectrum(energies, bandgap_ev, direct_bandgap_ev)
        # Beer-Lambert 定律算吸收率，乘以 2 考虑背电极的光反射 (光在薄膜中走两遍)
        absorptivity_aE = 1 - np.exp(-2 * alpha_E * L_cm)

        # 3. 计算短路电流 (Jsc)
        sun_flux = self._planck_photon_flux(energies, self.T_sun) * self.omega_sun / np.pi
        jsc = e * simpson(absorptivity_aE * sun_flux, x=energies) # A/m^2

        # 4. 计算反向饱和电流 (J0)
        # J0_r: 辐射复合电流
        cell_flux = self._planck_photon_flux(energies, self.T_cell)
        j0_r = e * np.pi * simpson(absorptivity_aE * cell_flux, x=energies)
        
        # fr: 辐射复合因子 (Yu et al. 的核心公式，惩罚间接带隙的非辐射复合)
        delta_E = direct_bandgap_ev - bandgap_ev
        fr = np.exp(-delta_E * e / (k * self.T_cell))
        
        # 总的 J0 包含非辐射复合
        j0 = j0_r / fr

        # 5. 求解 IV 曲线获得 Voc 和 最大功率点
        voc = (k * self.T_cell / e) * np.log(jsc / j0 + 1)

        def v_m_func(v):
            return jsc - j0 * (np.exp(e * v / (k * self.T_cell)) - 1) - \
                   (j0 * e * v / (k * self.T_cell)) * np.exp(e * v / (k * self.T_cell))
        
        try:
            v_m = fsolve(v_m_func, voc * 0.8)[0]
        except:
            v_m = voc * 0.8
            
        j_m = jsc - j0 * (np.exp(e * v_m / (k * self.T_cell)) - 1)
        p_max = v_m * j_m

        # 6. 总入射功率 (Pin)
        pin = simpson(energies * e * sun_flux, x=energies)

        # 7. 汇总输出
        pce = (p_max / pin) * 100
        ff = (p_max / (jsc * voc)) * 100

        # 判断材料属性
        material_type = "直接带隙" if delta_E < 0.01 else f"间接带隙 (能级差 $\Delta E$ = {delta_E:.2f} eV)"

        return {
            "bandgap_fundamental_eV": round(bandgap_ev, 3),
            "bandgap_direct_eV": round(direct_bandgap_ev, 3),
            "material_type": material_type,
            "film_thickness_um": thickness_um,
            "radiative_efficiency_fr": f"{fr:.2e}",
            "Jsc_mA_cm2": round(jsc / 10, 2),
            "Voc_V": round(voc, 3),
            "FF_percent": round(ff, 2),
            "SLME_PCE_percent": round(pce, 2),
            "scientific_note": "基于 SLME 理论计算。考虑了有限厚度导致的不完全光吸收，以及间接带隙引起的非辐射复合能量损失。SLME 效率始终低于理想的 S-Q 极限。"
        }

if __name__ == "__main__":
    expert = SLMEExpert()
    
    # 场景 1: 直接带隙材料 (比如优质的钙钛矿 MAPbI3)
    # 基础带隙和直接带隙重合，厚度 1 微米
    print("--- 直接带隙材料 (Eg=1.55 eV, L=1.0 um) ---")
    print(expert.calculate_slme(1.55, 1.55, thickness_um=1.0))

    # 场景 2: 间接带隙材料 (比如晶体硅 Si)
    # 基础带隙 1.12 eV, 但直接跃迁带隙高达 3.4 eV (非辐射复合极其严重)
    print("\n--- 间接带隙材料 (Si: Eg=1.12 eV, Eg_direct=3.4 eV, L=2.0 um) ---")
    print(expert.calculate_slme(1.12, 3.40, thickness_um=2.0))
