import os
import numpy as np
from scipy.integrate import simpson
from scipy.optimize import fsolve
from scipy.constants import h, c, k, e

class PVCalculatorExpert:
    """
    光伏性能计算专家。
    核心功能：基于详细平衡原理 (Detailed Balance) 计算材料的 Shockley-Queisser (S-Q) 理论极限。
    包含：短路电流 (Jsc)、开路电压 (Voc)、填充因子 (FF) 和 光电转换效率 (PCE)。
    """

    def __init__(self, T_cell=300, T_sun=5778):
        """
        :param T_cell: 电池工作温度 (标准状态下为 300K)
        :param T_sun: 太阳表面有效温度 (约 5778K)
        """
        self.T_cell = T_cell
        self.T_sun = T_sun
        # 太阳固体角 (从地球上看)
        self.omega_sun = 6.85e-5 
        
    def _planck_photon_flux(self, energy_ev, temp):
        """
        普朗克定律：计算给定能量和温度下的光子通量光谱密度。
        单位: photons / (s * m^2 * eV)
        """
        energy_j = energy_ev * e
        # 普朗克公式：B(E) = (2 * pi * E^2) / (h^3 * c^2 * (exp(E/kT) - 1))
        denominator = np.exp(energy_j / (k * temp)) - 1
        flux = (2 * np.pi * (energy_j**2)) / ((h**3) * (c**2) * denominator)
        return flux * e # 转化为每 eV 的通量

    def calculate_sq_limit(self, bandgap_ev: float):
        """
        核心计算：给定带隙，计算 S-Q 理论极限。
        :param bandgap_ev: 材料带隙 (eV)
        :return: 包含 PCE, Voc, Jsc, FF 的字典
        """
        if bandgap_ev <= 0:
            return {"error": "带隙必须大于 0 eV"}

        # 1. 能量网格 (0.01 eV 到 5.0 eV)
        energies = np.linspace(0.01, 5.0, 1000)
        mask = energies >= bandgap_ev
        active_energies = energies[mask]

        # 2. 计算短路电流 (Jsc)
        # 假设所有能量大于带隙的光子都被吸收 (QE=1)
        # 这里使用太阳黑体辐射近似 AM1.5G（科研中常用作标准对照）
        sun_flux = self._planck_photon_flux(active_energies, self.T_sun) * self.omega_sun / np.pi
        jsc = e * simpson(sun_flux, active_energies) # A/m^2

        # 3. 计算反向饱和电流 (J0) - 辐射复合损失
        # 根据详细平衡原理，电池在 300K 下也会向外辐射
        cell_flux = self._planck_photon_flux(active_energies, self.T_cell)
        j0 = e * simpson(cell_flux, active_energies) # A/m^2

        # 4. 计算开路电压 (Voc)
        # Voc = (kT/q) * ln(Jsc/J0 + 1)
        voc = (k * self.T_cell / e) * np.log(jsc / j0 + 1)

        # 5. 计算最大功率点和填充因子 (FF)
        # J(V) = Jsc - J0 * (exp(qV/kT) - 1)
        def v_m_func(v):
            return jsc - j0 * (np.exp(e * v / (k * self.T_cell)) - 1) - \
                   (j0 * e * v / (k * self.T_cell)) * np.exp(e * v / (k * self.T_cell))
        
        # 寻找使 dP/dV = 0 的 V_m
        v_m = fsolve(v_m_func, voc * 0.8)[0]
        j_m = jsc - j0 * (np.exp(e * v_m / (k * self.T_cell)) - 1)
        p_max = v_m * j_m

        # 6. 计算总入射功率 (Pin)
        all_sun_flux = self._planck_photon_flux(energies, self.T_sun) * self.omega_sun / np.pi
        pin = simpson(energies * e * all_sun_flux, energies) # W/m^2

        # 7. 汇总结果
        pce = (p_max / pin) * 100
        ff = (p_max / (jsc * voc)) * 100

        return {
            "bandgap_input_eV": round(bandgap_ev, 3),
            "Jsc_mA_cm2": round(jsc / 10, 2), # 转化为 mA/cm^2
            "Voc_V": round(voc, 3),
            "FF_percent": round(ff, 2),
            "PCE_percent": round(pce, 2),
            "note": "基于详细平衡原理和黑体辐射模型计算。"
        }

    def batch_screen(self, bandgap_list: list):
        """批量筛选不同带隙下的效率"""
        results = []
        for bg in bandgap_list:
            results.append(self.calculate_sq_limit(bg))
        return results

if __name__ == "__main__":
    # 单元测试
    calc = PVCalculatorExpert()
    
    # 场景 1: 测试单结硅电池理论效率 (带隙 1.12 eV)
    print("--- 硅电池 (1.12 eV) S-Q 极限测试 ---")
    si_res = calc.calculate_sq_limit(1.12)
    print(si_res)

    # 场景 2: 测试钙钛矿带隙 (1.55 eV)
    print("\n--- 钙钛矿 (1.55 eV) S-Q 极限测试 ---")
    ps_res = calc.calculate_sq_limit(1.55)
    print(ps_res)
