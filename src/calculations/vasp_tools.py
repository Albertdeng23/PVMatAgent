import os
import time
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import paramiko
    from paramiko import SSHClient, AutoAddPolicy
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    warnings.warn("paramiko not installed. VASP remote calculation features will be disabled.")

try:
    from ase.io import read, write
    from ase.calculators.vasp import Vasp
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    warnings.warn("ASE not installed. Structure conversion features will be limited.")

import numpy as np
import yaml


def load_config():
    """加载全局 YAML 配置文件"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@dataclass
class VASPConfig:
    """VASP计算配置"""
    xc: str = 'PBE'
    kpts: Tuple[int, int, int] = (4, 4, 4)
    encut: int = 400
    ismear: int = 0
    sigma: float = 0.05
    ibrion: int = 2
    nsw: int = 100
    ediff: float = 1e-6
    ediffg: float = -0.05
    isif: int = 2
    lwave: bool = True
    lcharg: bool = True
    
    def to_incar_dict(self) -> Dict:
        """转换为INCAR参数字典"""
        return {
            'ENCUT': self.encut,
            'ISMEAR': self.ismear,
            'SIGMA': self.sigma,
            'IBRION': self.ibrion,
            'NSW': self.nsw,
            'EDIFF': self.ediff,
            'EDIFFG': self.ediffg,
            'ISIF': self.isif,
            'LWAVE': '.TRUE.' if self.lwave else '.FALSE.',
            'LCHARG': '.TRUE.' if self.lcharg else '.FALSE.',
        }


@dataclass
class ServerConfig:
    """远程服务器配置"""
    hostname: str
    username: str
    password: Optional[str] = None
    key_filename: Optional[str] = None
    port: int = 22
    vasp_command: str = 'vasp_std'
    work_dir: str = '~/vasp_calculations'


class VASPExecutionError(Exception):
    """VASP执行错误"""
    pass


class VASPConnectionError(Exception):
    """VASP连接错误"""
    pass


class VASPToolsExpert:
    """
    VASP第一性原理计算专家。
    
    功能：
    1. 通过SSH连接远程超算服务器
    2. 自动准备VASP输入文件 (POSCAR, INCAR, KPOINTS, POTCAR)
    3. 提交和管理计算任务
    4. 实时监控计算进度
    5. 下载和解析结果文件
    
    ⚠️ 警告：VASP计算极其耗时（通常需要数小时到数天），
    且消耗大量计算资源。请仅在必要时使用此工具。
    """
    
    def __init__(self):
        print("⚛️ [VASP_Expert] Initializing VASP Tools...")
        
        if not PARAMIKO_AVAILABLE:
            print("❌ [VASP_Expert] paramiko not installed. Remote SSH features disabled.")
        if not ASE_AVAILABLE:
            print("❌ [VASP_Expert] ASE not installed. Structure conversion limited.")
        
        self.config = load_config()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.cache_dir = os.path.join(self.base_dir, "cache", "vasp_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.ssh_client: Optional[SSHClient] = None
        self.server_config: Optional[ServerConfig] = None
        self.current_job_id: Optional[str] = None
        
        print("✅ [VASP_Expert] VASP Tools initialized.")
    
    def connect_server(self, hostname: str, username: str, 
                      password: Optional[str] = None,
                      key_filename: Optional[str] = None,
                      port: int = 22,
                      vasp_command: str = 'vasp_std',
                      work_dir: str = '~/vasp_calculations') -> Dict:
        """
        连接远程VASP计算服务器。
        
        参数:
            hostname: 服务器地址 (IP或域名)
            username: 用户名
            password: 密码（与key_filename二选一）
            key_filename: SSH私钥路径（与password二选一）
            port: SSH端口，默认22
            vasp_command: VASP执行命令，默认'vasp_std'
            work_dir: 服务器上的工作目录
            
        返回:
            连接状态信息
        """
        if not PARAMIKO_AVAILABLE:
            return {"error": "paramiko库未安装，无法使用SSH功能。请运行: pip install paramiko"}
        
        try:
            self.ssh_client = SSHClient()
            self.ssh_client.set_missing_host_key_policy(AutoAddPolicy())
            
            connect_kwargs = {
                'hostname': hostname,
                'username': username,
                'port': port,
                'timeout': 30
            }
            
            if password:
                connect_kwargs['password'] = password
            elif key_filename:
                connect_kwargs['key_filename'] = key_filename
            else:
                return {"error": "必须提供password或key_filename之一"}
            
            self.ssh_client.connect(**connect_kwargs)
            
            # 测试连接
            stdin, stdout, stderr = self.ssh_client.exec_command('echo "Connection successful"')
            output = stdout.read().decode().strip()
            
            if output == "Connection successful":
                self.server_config = ServerConfig(
                    hostname=hostname,
                    username=username,
                    password=password,
                    key_filename=key_filename,
                    port=port,
                    vasp_command=vasp_command,
                    work_dir=work_dir
                )
                
                return {
                    "status": "success",
                    "message": f"成功连接到服务器 {hostname}",
                    "work_dir": work_dir,
                    "vasp_command": vasp_command
                }
            else:
                return {"error": "连接测试失败"}
                
        except paramiko.AuthenticationException:
            return {"error": "认证失败，请检查用户名和密码/密钥"}
        except paramiko.SSHException as e:
            return {"error": f"SSH连接错误: {str(e)}"}
        except Exception as e:
            return {"error": f"连接失败: {str(e)}"}
    
    def prepare_vasp_inputs(self, cif_path: str, config: Optional[VASPConfig] = None,
                           calculation_type: str = 'relax') -> Dict:
        """
        准备VASP输入文件。
        
        参数:
            cif_path: CIF文件路径
            config: VASP配置，默认使用标准配置
            calculation_type: 计算类型 ('relax', 'static', 'band')
            
        返回:
            输入文件路径字典
        """
        if not os.path.exists(cif_path):
            return {"error": f"CIF文件不存在: {cif_path}"}
        
        if config is None:
            config = VASPConfig()
        
        try:
            # 读取结构
            if ASE_AVAILABLE:
                structure = read(cif_path)
                formula = structure.get_chemical_formula()
            else:
                return {"error": "ASE未安装，无法读取CIF文件"}
            
            # 创建任务目录
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            job_name = f"{formula}_{calculation_type}_{timestamp}"
            job_dir = os.path.join(self.cache_dir, job_name)
            os.makedirs(job_dir, exist_ok=True)
            
            # 生成POSCAR
            poscar_path = os.path.join(job_dir, 'POSCAR')
            if ASE_AVAILABLE:
                write(poscar_path, structure, format='vasp')
            
            # 生成INCAR
            incar_path = os.path.join(job_dir, 'INCAR')
            incar_dict = config.to_incar_dict()
            
            # 根据计算类型调整参数
            if calculation_type == 'relax':
                incar_dict.update({'IBRION': 2, 'NSW': 100, 'ISIF': 3})
            elif calculation_type == 'static':
                incar_dict.update({'IBRION': -1, 'NSW': 0, 'ISIF': 2})
            elif calculation_type == 'band':
                incar_dict.update({
                    'IBRION': -1, 'NSW': 0, 'ISIF': 2,
                    'ICHARG': 11, 'LORBIT': 11
                })
            
            with open(incar_path, 'w') as f:
                f.write(f"# VASP input file for {formula}\n")
                f.write(f"# Calculation type: {calculation_type}\n")
                for key, value in incar_dict.items():
                    f.write(f"{key} = {value}\n")
            
            # 生成KPOINTS
            kpoints_path = os.path.join(job_dir, 'KPOINTS')
            with open(kpoints_path, 'w') as f:
                f.write("Automatic mesh\n")
                f.write("0\n")
                f.write("Gamma\n")
                f.write(f"{' '.join(map(str, config.kpts))}\n")
                f.write("0 0 0\n")
            
            return {
                "status": "success",
                "job_name": job_name,
                "job_dir": job_dir,
                "files": {
                    "POSCAR": poscar_path,
                    "INCAR": incar_path,
                    "KPOINTS": kpoints_path
                },
                "note": "请手动准备POTCAR文件（或使用pymatgen自动生成）"
            }
            
        except Exception as e:
            return {"error": f"准备输入文件失败: {str(e)}"}
    
    def submit_calculation(self, job_name: str, potcar_source: Optional[str] = None,
                          blocking: bool = False, timeout_hours: float = 24.0) -> Dict:
        """
        提交VASP计算到远程服务器。
        
        ⚠️ 警告：此操作将启动长时间计算任务，可能消耗大量计算资源！
        
        参数:
            job_name: 任务名称（由prepare_vasp_inputs生成）
            potcar_source: POTCAR文件来源路径（本地）
            blocking: 是否阻塞等待计算完成
            timeout_hours: 最大等待时间（小时）
            
        返回:
            提交状态和作业信息
        """
        if not self.ssh_client or not self.server_config:
            return {"error": "未连接到远程服务器，请先调用connect_server"}
        
        job_dir = os.path.join(self.cache_dir, job_name)
        if not os.path.exists(job_dir):
            return {"error": f"任务目录不存在: {job_dir}"}
        
        try:
            # 检查必需文件
            required_files = ['POSCAR', 'INCAR', 'KPOINTS']
            for f in required_files:
                if not os.path.exists(os.path.join(job_dir, f)):
                    return {"error": f"缺少必需文件: {f}"}
            
            # 处理POTCAR
            if potcar_source and os.path.exists(potcar_source):
                import shutil
                shutil.copy(potcar_source, os.path.join(job_dir, 'POTCAR'))
            elif not os.path.exists(os.path.join(job_dir, 'POTCAR')):
                return {
                    "error": "缺少POTCAR文件",
                    "note": "请提供potcar_source路径或手动放置POTCAR文件到任务目录"
                }
            
            # 创建远程工作目录
            remote_job_dir = f"{self.server_config.work_dir}/{job_name}"
            stdin, stdout, stderr = self.ssh_client.exec_command(f'mkdir -p {remote_job_dir}')
            stdout.channel.recv_exit_status()
            
            # 上传文件
            sftp = self.ssh_client.open_sftp()
            local_files = ['POSCAR', 'INCAR', 'KPOINTS', 'POTCAR']
            
            print(f"📤 [VASP] 上传输入文件到 {remote_job_dir}...")
            for filename in local_files:
                local_path = os.path.join(job_dir, filename)
                remote_path = f"{remote_job_dir}/{filename}"
                sftp.put(local_path, remote_path)
            sftp.close()
            
            # 提交计算
            vasp_cmd = self.server_config.vasp_command
            submit_cmd = f"cd {remote_job_dir} && {vasp_cmd} > vasp.out 2>&1 & echo $!"
            
            print(f"🚀 [VASP] 提交计算任务...")
            stdin, stdout, stderr = self.ssh_client.exec_command(submit_cmd)
            pid = stdout.read().decode().strip()
            self.current_job_id = pid
            
            result = {
                "status": "submitted",
                "job_name": job_name,
                "remote_dir": remote_job_dir,
                "pid": pid,
                "warning": "VASP计算可能需要数小时到数天，请使用check_status查询进度"
            }
            
            if blocking:
                print(f"⏳ [VASP] 等待计算完成（最长{timeout_hours}小时）...")
                start_time = time.time()
                timeout_seconds = timeout_hours * 3600
                
                while True:
                    time.sleep(30)  # 每30秒检查一次
                    status = self.check_status(job_name)
                    
                    if status.get("status") == "completed":
                        result["completion_status"] = "success"
                        result["elapsed_time"] = time.time() - start_time
                        break
                    elif status.get("status") == "failed":
                        result["completion_status"] = "failed"
                        result["error"] = status.get("error", "Unknown error")
                        break
                    
                    if time.time() - start_time > timeout_seconds:
                        result["completion_status"] = "timeout"
                        result["warning"] = f"计算超时（>{timeout_hours}小时），任务仍在后台运行"
                        break
            
            return result
            
        except Exception as e:
            return {"error": f"提交计算失败: {str(e)}"}
    
    def check_status(self, job_name: str) -> Dict:
        """
        检查计算任务状态。
        
        参数:
            job_name: 任务名称
            
        返回:
            任务状态信息
        """
        if not self.ssh_client or not self.server_config:
            return {"error": "未连接到远程服务器"}
        
        remote_job_dir = f"{self.server_config.work_dir}/{job_name}"
        
        try:
            # 检查OUTCAR是否存在且包含正常结束标志
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f'grep "Total CPU time used" {remote_job_dir}/OUTCAR 2>/dev/null | tail -1'
            )
            outcar_check = stdout.read().decode().strip()
            
            if outcar_check and "Total CPU time used" in outcar_check:
                return {
                    "status": "completed",
                    "job_name": job_name,
                    "message": "计算已完成",
                    "note": "可以使用download_results下载结果文件"
                }
            
            # 检查vasp.out是否存在
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f'ls -la {remote_job_dir}/vasp.out 2>/dev/null'
            )
            vasp_out_exists = stdout.read().decode().strip()
            
            if not vasp_out_exists:
                return {
                    "status": "pending",
                    "job_name": job_name,
                    "message": "任务正在排队或尚未启动"
                }
            
            # 检查进程是否仍在运行
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f'ps -p {self.current_job_id} -o pid,etime,%cpu 2>/dev/null | tail -1'
            )
            process_info = stdout.read().decode().strip()
            
            if process_info:
                return {
                    "status": "running",
                    "job_name": job_name,
                    "process_info": process_info,
                    "message": "计算正在进行中"
                }
            else:
                # 进程结束但OUTCAR不完整，可能出错
                stdin, stdout, stderr = self.ssh_client.exec_command(
                    f'tail -20 {remote_job_dir}/vasp.out'
                )
                last_lines = stdout.read().decode().strip()
                
                return {
                    "status": "failed",
                    "job_name": job_name,
                    "error": "计算进程已结束但未正常完成",
                    "last_output": last_lines
                }
                
        except Exception as e:
            return {"error": f"检查状态失败: {str(e)}"}
    
    def download_results(self, job_name: str, output_files: Optional[List[str]] = None) -> Dict:
        """
        下载计算结果文件。
        
        参数:
            job_name: 任务名称
            output_files: 要下载的文件列表，默认下载所有重要文件
            
        返回:
            下载状态信息
        """
        if not self.ssh_client or not self.server_config:
            return {"error": "未连接到远程服务器"}
        
        if output_files is None:
            # 默认下载的重要文件
            output_files = ['OUTCAR', 'vasprun.xml', 'CONTCAR', 'EIGENVAL', 'DOSCAR', 
                          'vasp.out', 'CHGCAR', 'WAVECAR']
        
        remote_job_dir = f"{self.server_config.work_dir}/{job_name}"
        local_job_dir = os.path.join(self.cache_dir, job_name, 'results')
        os.makedirs(local_job_dir, exist_ok=True)
        
        try:
            sftp = self.ssh_client.open_sftp()
            downloaded_files = []
            failed_files = []
            
            print(f"📥 [VASP] 下载结果文件到 {local_job_dir}...")
            
            for filename in output_files:
                remote_path = f"{remote_job_dir}/{filename}"
                local_path = os.path.join(local_job_dir, filename)
                
                try:
                    sftp.get(remote_path, local_path)
                    downloaded_files.append(filename)
                except Exception as e:
                    failed_files.append(f"{filename}: {str(e)}")
            
            sftp.close()
            
            return {
                "status": "success",
                "job_name": job_name,
                "local_dir": local_job_dir,
                "downloaded_files": downloaded_files,
                "failed_files": failed_files if failed_files else None,
                "note": "结果文件已下载到本地缓存目录"
            }
            
        except Exception as e:
            return {"error": f"下载结果失败: {str(e)}"}
    
    def parse_eigenval(self, job_name: str) -> Dict:
        """
        解析EIGENVAL文件获取能带结构数据。
        
        参数:
            job_name: 任务名称
            
        返回:
            能带结构数据
        """
        local_job_dir = os.path.join(self.cache_dir, job_name, 'results')
        eigenval_path = os.path.join(local_job_dir, 'EIGENVAL')
        
        if not os.path.exists(eigenval_path):
            return {"error": f"EIGENVAL文件不存在: {eigenval_path}"}
        
        try:
            with open(eigenval_path, 'r') as f:
                lines = f.readlines()
            
            # 读取头部信息（第6行）
            header = lines[5].split()
            num_electrons = int(header[0])
            num_kpoints = int(header[1])
            num_bands = int(header[2])
            
            # 读取能带数据（跳过前8行）
            energies = []
            kpoints = []
            
            line_idx = 8
            for _ in range(num_kpoints):
                # 读取k点坐标和权重
                kpt_line = lines[line_idx].split()
                kx, ky, kz = float(kpt_line[0]), float(kpt_line[1]), float(kpt_line[2])
                weight = float(kpt_line[3])
                kpoints.append([kx, ky, kz, weight])
                line_idx += 1
                
                # 读取该k点的所有能带能量
                band_energies = []
                for _ in range(num_bands):
                    energy = float(lines[line_idx].split()[1])  # 第2列是能量
                    band_energies.append(energy)
                    line_idx += 1
                
                energies.append(band_energies)
            
            # 转换为numpy数组便于处理
            energies_array = np.array(energies)  # shape: (num_kpoints, num_bands)
            
            # 估算费米能级（取最高占据能带的中值）
            # 对于金属，这可能不准确
            half_electrons = num_electrons / 2
            homo_band_idx = int(half_electrons) - 1
            lumo_band_idx = homo_band_idx + 1
            
            efermi_estimate = np.mean([
                np.max(energies_array[:, homo_band_idx]),
                np.min(energies_array[:, lumo_band_idx])
            ]) if lumo_band_idx < num_bands else np.max(energies_array[:, homo_band_idx])
            
            return {
                "status": "success",
                "job_name": job_name,
                "num_electrons": num_electrons,
                "num_kpoints": num_kpoints,
                "num_bands": num_bands,
                "efermi_estimate": round(float(efermi_estimate), 4),
                "band_gap_estimate": self._estimate_band_gap(energies_array, efermi_estimate),
                "kpoints": kpoints[:5] if len(kpoints) > 5 else kpoints,  # 只返回前5个k点示例
                "note": "能带数据已解析， energies数组大小为 (num_kpoints, num_bands)"
            }
            
        except Exception as e:
            return {"error": f"解析EIGENVAL失败: {str(e)}"}
    
    def _estimate_band_gap(self, energies: np.ndarray, efermi: float) -> Dict:
        """估算带隙"""
        num_kpoints, num_bands = energies.shape
        
        # 找到VBM和CBM
        vbm = -np.inf
        vbm_kpoint = -1
        cbm = np.inf
        cbm_kpoint = -1
        
        for k_idx in range(num_kpoints):
            for band_idx in range(num_bands):
                energy = energies[k_idx, band_idx]
                if energy <= efermi and energy > vbm:
                    vbm = energy
                    vbm_kpoint = k_idx
                elif energy > efermi and energy < cbm:
                    cbm = energy
                    cbm_kpoint = k_idx
        
        band_gap = cbm - vbm if cbm > vbm else 0.0
        
        return {
            "band_gap_eV": round(float(band_gap), 4),
            "vbm_eV": round(float(vbm), 4),
            "cbm_eV": round(float(cbm), 4),
            "is_direct": vbm_kpoint == cbm_kpoint,
            "vbm_kpoint": int(vbm_kpoint),
            "cbm_kpoint": int(cbm_kpoint)
        }
    
    def disconnect(self) -> Dict:
        """断开服务器连接"""
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
            self.server_config = None
            return {"status": "disconnected", "message": "已断开服务器连接"}
        return {"status": "not_connected", "message": "当前未连接服务器"}


if __name__ == "__main__":
    # 单元测试
    expert = VASPToolsExpert()
    
    print("\n=== VASP Tools Expert 测试 ===")
    print("可用功能:")
    print("1. connect_server() - 连接远程服务器")
    print("2. prepare_vasp_inputs() - 准备输入文件")
    print("3. submit_calculation() - 提交计算")
    print("4. check_status() - 检查状态")
    print("5. download_results() - 下载结果")
    print("6. parse_eigenval() - 解析能带数据")
    print("7. disconnect() - 断开连接")
    print("\n⚠️ 注意：需要配置远程服务器才能进行实际计算")
