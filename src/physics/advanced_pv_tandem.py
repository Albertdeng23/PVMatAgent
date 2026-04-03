import numpy as np
import warnings
from scipy.integrate import simpson
from scipy.constants import h, c, k, e

warnings.filterwarnings("ignore")

class TandemPVExpert:
    """
    高阶光伏物理专家：两端 (2-Terminal) 叠层太阳能电池理论极限与电流匹配计算器。
    基于详细平衡原理，计算串联结构下顶电池(Top Cell)和底电池(Bottom Cell)的光谱分配，
    并精确求解电流失配限制下的总效率。
    """
    def __init__(self, T_cell=300, T_sun=5778):
        self.T_cell = T_cell
        self.T_sun = T_sun
        self.omega_sun = 6.85e-5 # 太阳固体角
        print("🔋 [Advanced_PV_Expert] Initializing 2T Tandem Solar Cell Simulator...")

    def _planck_photon_flux(self, energy_ev, temp):
        """计算普朗克黑体辐射光子通量密度 (photons / s*m^2*eV)"""
        energy_j = energy_ev * e
        exponent = np.clip(energy_j / (k * temp), a_min=None, a_max=700)
        denominator = np.exp(exponent) - 1
        flux = (2 * np.pi * (energy_j**2)) / ((h**3) * (c**2) * denominator)
        return flux * e

    def calculate_2t_tandem(self, bandgap_top_ev: float, bandgap_bot_ev: float):
        """
        核心计算：两端串联叠层电池理论极限。
        :param bandgap_top_ev: 顶电池带隙 (宽带隙，吸收高能光子)
        :param bandgap_bot_ev: 底电池带隙 (窄带隙，吸收低能光子)
        """
        if bandgap_top_ev <= bandgap_bot_ev:
            return {"error": "顶电池带隙必须严格大于底电池带隙！(高能光子必须先被顶层吸收)"}
        
        # 1. 建立能量网格
        energies = np.linspace(0.1, 5.0, 5000)
        sun_flux = self._planck_photon_flux(energies, self.T_sun) * self.omega_sun / np.pi
        cell_flux = self._planck_photon_flux(energies, self.T_cell)
        
        # 2. 计算顶电池 (Top Cell)
        # 顶电池吸收大于 Eg_top 的所有光子
        mask_top = energies >= bandgap_top_ev
        jsc_top = e * simpson(sun_flux[mask_top], x=energies[mask_top])
        j0_top = e * np.pi * simpson(cell_flux[mask_top], x=energies[mask_top])
        
        # 3. 计算底电池 (Bottom Cell)
        # 底电池只能吸收透过顶电池的光子，即能量在 [Eg_bot, Eg_top) 之间的光子
        mask_bot = (energies >= bandgap_bot_ev) & (energies < bandgap_top_ev)
        jsc_bot = e * simpson(sun_flux[mask_bot], x=energies[mask_bot])
        # 底电池的热辐射范围是大于 Eg_bot 的所有区域
        mask_bot_emit = energies >= bandgap_bot_ev
        j0_bot = e * np.pi * simpson(cell_flux[mask_bot_emit], x=energies[mask_bot_emit])
        
        # 4. 计算串联器件参数 (2-Terminal Constraint)
        # 串联电流必须相等，理论短路电流受限于最小者
        jsc_tandem = min(jsc_top, jsc_bot)
        
        # 电流失配量 (mA/cm^2)
        mismatch_mA = abs(jsc_top - jsc_bot) / 10.0
        
        # 寻找最大功率点 (MPP)
        # 对串联器件：V_total(J) = V_top(J) + V_bot(J), 需要最大化 P = J * V_total(J)
        # J 从 0 扫描到 jsc_tandem
        J_scan = np.linspace(0, jsc_tandem * 0.9999, 1000)
        
        def v_from_j(j, jsc, j0):
            val = np.maximum((jsc - j) / j0 + 1, 1e-12)
            return (k * self.T_cell / e) * np.log(val)
            
        V_top_scan = v_from_j(J_scan, jsc_top, j0_top)
        V_bot_scan = v_from_j(J_scan, jsc_bot, j0_bot)
        V_total_scan = V_top_scan + V_bot_scan
        
        P_scan = J_scan * V_total_scan
        max_idx = np.argmax(P_scan)
        
        p_max = P_scan[max_idx]
        v_mpp = V_total_scan[max_idx]
        j_mpp = J_scan[max_idx]
        
        # 开路电压 (J=0)
        voc_tandem = V_total_scan[0]
        voc_top = V_top_scan[0]
        voc_bot = V_bot_scan[0]
        
        # 总入射太阳功率
        pin = simpson(energies * e * sun_flux, x=energies)
        
        # 效率与填充因子
        pce = (p_max / pin) * 100
        ff = (p_max / (jsc_tandem * voc_tandem)) * 100
        
        # 生成科学建议
        if mismatch_mA > 1.0:
            if jsc_top > jsc_bot:
                advice = "严重电流失配。底电池电流不足成为瓶颈。建议【增大顶电池带隙】以让出更多光子给底电池，或【减小底电池带隙】。"
            else:
                advice = "严重电流失配。顶电池电流不足成为瓶颈。建议【减小顶电池带隙】以吸收更多光子。"
        else:
            advice = "极佳的电流匹配组合！两子电池电流高度一致，能发挥出 2T 叠层结构的最大潜力。"

        return {
            "bandgap_combination_eV": {"top": bandgap_top_ev, "bottom": bandgap_bot_ev},
            "subcell_Voc_V": {"top": round(voc_top, 3), "bottom": round(voc_bot, 3)},
            "subcell_Jsc_mA_cm2": {
                "top_generated": round(jsc_top / 10, 2), 
                "bottom_generated": round(jsc_bot / 10, 2)
            },
            "mismatch_mA_cm2": round(mismatch_mA, 2),
            "tandem_performance": {
                "Jsc_mA_cm2": round(jsc_tandem / 10, 2),
                "Voc_V": round(voc_tandem, 3),
                "FF_percent": round(ff, 2),
                "PCE_percent": round(pce, 2)
            },
            "scientific_advice": advice
        }

if __name__ == "__main__":
    expert = TandemPVExpert()
    
    # 场景 1: 经典的 钙钛矿/晶硅 叠层 (顶层最佳带隙约 1.73 eV，底硅 1.12 eV)
    print("--- 钙钛矿/硅 叠层 (1.73 eV / 1.12 eV) ---")
    res1 = expert.calculate_2t_tandem(1.73, 1.12)
    import json
    print(json.dumps(res1, indent=2, ensure_ascii=False))

    # 场景 2: 匹配极差的组合 (顶层带隙太小，把光吸完了，底电池没光可用)
    print("\n--- 错误匹配组合 (1.50 eV / 1.12 eV) ---")
    res2 = expert.calculate_2t_tandem(1.50, 1.12)
    print(res2['scientific_advice'])
