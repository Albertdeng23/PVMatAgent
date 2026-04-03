# 🌞 MatMoE-Agent: 基于大模型与多专家协同的光伏材料智能发现系统
**(Photovoltaic Materials Mixture of Experts Agent)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Tool_Calling-green)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Materials Project](https://img.shields.io/badge/Data-Materials_Project-orange)](https://materialsproject.org/)
[![Status: Stable](https://img.shields.io/badge/Status-Stable-brightgreen)]()

MatMoE-Agent 是一个专为**光伏材料（尤其是无铅/卤化物钙钛矿）研发**设计的智能科学助手（AI for Science Agent）。系统创新性地采用了**“神经-符号（Neuro-symbolic）”**协同范式，以云端大语言模型（LLMs）为逻辑中枢，通过原生函数调用（Tool Calling）架构，无缝调度底层的 13 项包含物理计算、深度学习预测、化学数据库与工程评估的“专家工具”。

该系统彻底消除了大语言模型在垂直科学领域严重的“数值幻觉”，实现了从“自然语言需求输入”到“多维度自动化科研报告输出”的全链路材料发现与诊断闭环。

---

## 📑 目录 (Table of Contents)
1. [项目背景与痛点解决](#1-项目背景与痛点解决)
2. [系统核心架构设计](#2-系统核心架构设计)
3. [十三大超级专家工具矩阵](#3-十三大超级专家工具矩阵)
4. [项目安装与配置指南](#4-项目安装与配置指南)
5. [系统使用与交互示例](#5-系统使用与交互指南)
6. [典型科研场景案例 (Showcases)](#6-典型科研场景案例-showcases)
7. [技术亮点与疑难 Bug 修复方案](#7-技术亮点与底层-bug-修复)
8. [项目目录结构](#8-项目目录结构)
9. [未来展望 (Roadmap)](#9-未来展望-roadmap)
10. [致谢与参考文献](#10-致谢与参考文献)

---

## 1. 项目背景与痛点解决

### 1.1 传统材料计算的瓶颈
在寻找无铅、稳定且高效的新型光伏候选材料时，传统的研究范式主要依赖于基于密度泛函理论（DFT）的高通量计算。然而，DFT 计算在面对包含数百个原子的复杂晶胞、结构弛豫或缺陷过渡态搜索时，其算力消耗呈指数级增长，导致材料筛选周期漫长。

### 1.2 通用大模型（LLMs）的“科学偏科”
通用大语言模型（如 GPT-4, Qwen, DeepSeek）拥有极强的逻辑推理能力，但在材料科学领域面临三大致命缺陷：
1. **数值幻觉**：容易凭空捏造带隙、形成能等严谨的物理参数。
2. **三维盲区**：无法解析或操作高维的三维晶体结构数据（如 CIF 文件）。
3. **缺乏算力**：无法执行复杂的微积分运算（如 S-Q 极限黑体辐射积分）。

### 1.3 MatMoE-Agent 的破局之道
本系统将 LLM 降维为**“任务调度器（Planner）”**，将所有严谨的物理计算、查表评估和结构操作交由底层的 Python 专家模块（Expert Tools）执行。LLM 依靠系统内置的**标准作业程序（SOP Prompt）**，自主决定工具的调用顺序、参数传递与异常纠错，最终生成图文并茂的 Markdown 综合科研报告。

---

## 2. 系统核心架构设计

本系统全面拥抱了面向云端大模型的 **Tool Calling（原生函数调用）架构**，替代了早期脆弱的 ReAct 文本截断解析模式。

*   **大脑层 (Cognitive Brain)**：支持通过 OpenAI 兼容 API 接入任意顶尖云端模型（如 DeepSeek-V3, GPT-4o, Qwen-Max）。
*   **中枢层 (Orchestrator)**：基于 LangChain 的 `AgentExecutor`，管理工具注册、消息上下文缓冲（Scratchpad）以及并发工具调用（Parallel Tool Calling）。
*   **专家层 (Expert Modules)**：高度解耦的 Python 模块库，分为知识检索、底层物理、高阶光伏、工程合成四大编队。
*   **数据层 (Data & Configs)**：将有机阳离子半径、化工原料价格、汉森溶解度参数等硬编码数据抽离为 `YAML/JSON` 静态库，保障学术严谨性与系统扩展性。

---

## 3. 十三大超级专家工具矩阵

系统内建 13 个原子级工具，覆盖材料科学的研发全生命周期：

### 📖 模块一：知识与数据库专家 (Knowledge & Database)
1. **`tool_rag_search` (文献检索专家)**：基于 ChromaDB 的 Hybrid（BM25+向量）双路召回与 Cross-Encoder 重排序引擎。当计算工具失效或遇到带隙实验值等未知参数时，Agent 自动调用此工具溯源至 3000 篇光伏核心文献。
2. **`tool_structure_search` (结构下载专家)**：对接 Materials Project (MP) API，自动规范化有机缩写（如 MA, FA），根据热力学稳定性（Energy Above Hull）智能下载最稳定的 CIF 晶体结构。

### ⚛️ 模块二：底层物理与光学专家 (Physics & Optics)
3. **`tool_property_calculation` (热力学稳定性专家)**：挂载 **CHGNet** 预训练万能图神经网络力场。支持毫秒级的晶格弛豫（Relaxation）与形成能（Formation Energy）预测，一票否决热力学不稳定的配方。
4. **`tool_bandgap_predictor` (高精度带隙预测专家)**：挂载 **MEGNet** 深度图神经网络。从纯几何 CIF 结构直接预测 PBE 级别的材料带隙，自带底层 Cython 内存对齐防崩溃补丁。
5. **`tool_electronic_band_analyzer` (能带结构与有效质量专家)**：通过 MP 数据库解析高精度能带结构，提取导带底（CBM）与价带顶（VBM）的能量状态。

### ☀️ 模块三：高阶光伏物理专家 (Advanced Photovoltaics)
6. **`tool_pv_performance_calc` (S-Q 极限光伏效率专家)**：基于 SciPy 数值积分与详细平衡原理，计算单结电池的理论极限 $PCE, V_{oc}, J_{sc}, FF$。
7. **`tool_slme_efficiency_calculator` (SLME 光谱限制最大效率专家)**：引入 Tauc 模型与薄膜厚度限制。精确评估**间接带隙材料（如晶硅）**因非辐射复合导致的严重效率衰减，突破 S-Q 极限的理想化假设。
8. **`tool_tandem_current_matcher` (两端 2T 叠层电池匹配专家)**：求解宽带隙顶电池与窄带隙底电池串联时的光谱分配。自动计算电流失配度（Mismatch），并给出针对性的带隙组分优化建议。

### 🏭 模块四：工程、商业与实验专家 (Engineering & Synthesis)
9. **`tool_goldschmidt_tolerance` (容差因子快筛专家)**：调用 Pymatgen 权威 Shannon 半径库与外部有机离子库（如 MA, FA），极速计算 $ABX_3$ 钙钛矿的容差因子 $t$ 与八面体因子 $\mu$，评估其 3D 成相潜力。
10. **`tool_commercial_assessment` (商业成本与毒性环保专家)**：解析化学式摩尔质量，查阅大宗商品静态库估算合成 1kg 材料的原料成本（USD），并基于 RoHS 指令筛查剧毒重金属（Pb, Cd, Hg）。
11. **`tool_solvent_recommendation` (实验合成与溶剂推荐专家)**：基于汉森溶解度参数（HSP）计算三维空间距离 $R_a$，自动为前驱体（如 $PbI_2$）推荐用于旋涂工艺的最佳主溶剂与反溶剂组合。

### ✨ 模块五：新材料自主发现专家 (Material Discovery)
12. **`tool_generate_by_substitution` (元素替换新材料生成器)**：执行同构元素替换（如 "Pb:Sn, I:Br"），基于母体 CIF 生成人类尚未合成的全新假设材料。
13. **`tool_generate_cubic_perovskite` (理想立方钙钛矿生成器)**：基于硬球模型经验半径，从零构建标准的 Pm-3m 立方晶胞初态。*(注：生成后 Agent 会被 Prompt 强制要求调用 CHGNet 进行几何弛豫)*。

---

## 4. 项目安装与配置指南

### 4.1 环境要求
*   **OS**: Windows 10/11, Linux, or macOS.
*   **Python**: >= 3.9 (推荐使用 Anaconda 创建独立虚拟环境)
*   **Hardware**: 推荐具备 NVIDIA GPU 以加速 CHGNet/MEGNet 的推理，CPU 亦可运行但稍慢。

### 4.2 安装步骤
```bash
# 1. 克隆代码仓库
git clone https://github.com/your-username/MatMoE-Agent.git
cd MatMoE-Agent

# 2. 创建虚拟环境
conda create -n matagent python=3.10
conda activate matagent

# 3. 安装极其严苛的依赖版本 (防止 LangChain / Keras 冲突)
pip install langchain==0.3.0 langchain-core==0.3.63 langchain-community==0.3.0 langchain-openai==0.2.2 
pip install httpx==0.27.2 openai==1.52.0
pip install pymatgen chgnet megnet scipy pyyaml python-dotenv
```

### 4.3 全局配置 (`configs/config.yaml`)
检查 `configs/config.yaml` 文件，确保模型模式设置为 `cloud`。
```yaml
system:
  cache_dir: "./cache/cif_cache"
  log_dir: "./logs"

model:
  type: "cloud"  # 开启云端大模型 Tool Calling
  cloud_api:
    base_url: "https://api.deepseek.com/v1" # 你的大模型服务商地址
    model_name: "deepseek-chat"             # 模型名称
  temperature: 0.01
```

### 4.4 环境变量配置 (`.env`)
在项目根目录下新建一个 `.env` 文件，填入你的 API 密钥：
```env
# 必须配置：大模型服务商的 API Key (如 OpenAI, DeepSeek, 阿里云等)
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 必须配置：Materials Project 的 API Key (免费去官网注册获取)
MP_API_KEY=YOUR_MP_API_KEY
```

---

## 5. 系统使用与交互指南

配置完成后，在项目根目录运行主程序：
```bash
python main.py
```
系统初始化完毕后，你将看到交互式终端。你可以使用接近人类的**真实自然语言（Non-directive）**向它提问。

**⚠️ 重要特性：自动 Markdown 报告导出**
对于你的每一个问题，系统不仅会在终端输出简报，还会自动在 `logs/reports/` 目录下生成一份名为 `MatReport_YYYYMMDD_HHMMSS.md` 的详细科研报告。该报告包含了：
1. **最终研究报告 (Final Report)**：专业排版的学术结论。
2. **智能体工具调用轨迹 (Agent Trace)**：Agent 思考和调用工具的完整 JSON 输入输出快照，保证计算全过程**严谨、可复现、可溯源**。

---

## 6. 典型科研场景案例 (Showcases)

系统具备真正的“零样本自主规划（Zero-shot Planning）”能力。以下为四个典型的测试指令示例，你可以直接复制使用：

### 🧪 场景一：实验立项的综合快筛（无结构并发评估）
> **指令**：`“我准备在实验室立项做无铅的 FASnI3 太阳能电池。在买试剂动手之前，你能帮我全面评估一下这个材料的综合可行性吗？包括 3D 成相潜力、原料成本、环保属性以及推荐的旋涂溶剂工艺。”`
> 
> **Agent 行为**：并发调用 `tool_goldschmidt_tolerance`、`tool_commercial_assessment` 和 `tool_solvent_recommendation`。瞬间毙掉高价配方或给出毒性警告，并输出具体的 HSP 溶剂匹配数据。

### 🔭 场景二：揭穿物理陷阱（高阶 SLME 对比）
> **指令**：`“为了节省硅料成本，我想把工业上的晶硅做成 2 微米厚的超薄柔性太阳能电池。你觉得这个方案靠谱吗？请用 SLME 工具评估它，并和同样是 1 微米厚的典型钙钛矿 MAPbI3 相比，解释硅为何不能做成薄膜电池的物理原因。”`
> 
> **Agent 行为**：自动查询硅与 MAPbI3 的基础/直接带隙参数，调用 `tool_slme_efficiency_calculator`。系统将用数学数据向你证明：硅由于间接带隙导致严重的非辐射复合，2 微米厚度的 SLME 效率极其低下，从而否决该错误方案。

### 🔋 场景三：顶刊级器件系统设计（叠层电池匹配诊断）
> **指令**：`“我想研发一款钙钛矿-硅两端叠层电池。为了图省事，我准备直接拿目前最成熟的 MAPbI3 盖在晶硅底电池上。这个器件设计方案合理吗？如果不合理，我该怎么改进钙钛矿层？”`
> 
> **Agent 行为**：调用 `tool_tandem_current_matcher`。发现 MAPbI3（1.55eV）漏给底电池的光子太少，产生严重“电流失配”。Agent 根据物理工具返回的诊断，自主建议在顶层掺入 Br 元素扩大带隙至 1.73 eV 以实现串联匹配。

### ✨ 场景四：新材料的高通量闭环发现（终极盲测）
> **指令**：`“现有的 Cs2AgBiBr6 带隙有点大。请以它为母体，利用元素替换工具将溴全部换成氯，探索一种新材料。务必对生成的新结构进行几何弛豫（relax），随后严谨地评估新材料的热力学稳定性及其带隙变化趋势。”`
> 
> **Agent 行为**：
> 1. 下载 `Cs2AgBiBr6`。
> 2. 本地生成 `AI_Generated_Cs2AgBiCl6.cif`。
> 3. **听从 SOP 约束**：强制调用 CHGNet 进行几何弛豫寻找能量极小值。
> 4. 调用 MEGNet 预测新带隙，并在报告中得出“由于 Cl 的电负性强于 Br，材料带隙发生蓝移（变宽）”的专业学术结论。

---

## 7. 技术亮点与底层 Bug 修复

本项目的极客精神体现在对底层工具的深度打磨上：

### 🛠️ MEGNet Windows 兼容性终极补丁
在 Windows 下，Numpy 默认的 `int` 是 32 位，这会导致 MEGNet 调用底层 Pymatgen 的 C++ (Cython) 扩展模块寻找邻居原子时触发严重的 `Buffer dtype mismatch (expected int64_t)` 内存对齐崩溃。
**解决方案**：MatMoE-Agent 在 `electronic_optics.py` 初始化阶段实施了精密的**命名空间级劫持（Monkey Patch）**：
```python
# 暴力替换 MEGNet 命名空间中的引用，拦截传入 Cython 的数组并强转 int64
import megnet.utils.data as megnet_data
orig_find_points = megnet_data.find_points_in_spheres
def patched_find_points(all_coords, center_coords, r, pbc, lattice, tol, min_r):
    pbc_int64 = np.array(pbc, dtype=np.int64)
    return orig_find_points(all_coords, center_coords, r, pbc_int64, lattice, tol, min_r)
megnet_data.find_points_in_spheres = patched_find_points
```

### 🛡️ 高度容错的 Fallback 机制
由于新生成材料的复杂性，力场和机器学习预测极易失败（例如 MEGNet 抛出异常）。此时大模型绝不会使整个程序崩溃退出，而是会立刻打印出异常原因，并在 Prompt 的约束下**自主调用 RAG 检索工具**去顶刊文献中查阅替代数据，保证科研逻辑链不中断。

---

## 8. 项目目录结构

```text
MatMoE_Agent/
├── configs/                      # [静态数据与配置层]
│   ├── config.yaml               # 全局模型与系统参数
│   ├── hsp_database.yaml         # 汉森溶解度参数数据库
│   ├── materials_cost_toxicity.yaml # 大宗商品价格与 RoHS 毒性库
│   └── organic_radii.yaml        # 有机大阳离子有效半径文献库
├── data/                         # [数据层]
│   └── vector_db/                # ChromaDB 向量数据库 (存储3000篇文献)
├── src/                          # [核心逻辑代码]
│   ├── agent/
│   │   ├── brain.py              # LLM 底座接口 (兼容 Local/Cloud)
│   │   ├── orchestrator.py       # 智能体中枢 (Tool Calling 核心调度器)
│   │   └── prompts.py            # 面向科研工作流的 SOP Prompt
│   ├── database/
│   │   └── mp_handler.py         # Materials Project 下载专家
│   ├── discovery/
│   │   └── crystal_generator.py  # 元素替换与新晶体创造专家
│   ├── engineering/
│   │   └── techno_economic.py    # 成本估算与毒性环保专家
│   ├── knowledge/
│   │   └── rag_engine.py         # 混合双路召回 RAG 专家
│   ├── physics/
│   │   ├── advanced_pv_slme.py   # SLME 光谱限制最大效率专家
│   │   ├── advanced_pv_tandem.py # 两端(2T)叠层电池匹配专家
│   │   ├── electronic_optics.py  # MEGNet带隙与能带结构专家
│   │   ├── pv_calculators.py     # S-Q 极限黑体辐射专家
│   │   ├── stability.py          # CHGNet 几何弛豫与形成能专家
│   │   └── thermo_kinetics.py    # 钙钛矿容差因子快筛专家
│   └── synthesis/
│       └── lab_assistant.py      # HSP 溶剂匹配与工艺推荐专家
├── cache/
│   └── cif_cache/                # 自动保存/生成的 CIF 文件池
├── logs/
│   └── reports/                  # 自动导出的 Markdown 综合科研报告
├── .env                          # API 密钥管理 (Git忽略)
└── main.py                       # 极简 CLI 终端启动入口
```

---

## 9. 未来展望 (Roadmap)
*   **可视化 Web UI**：计划引入 Gradio 或 Streamlit，将纯终端的交互转化为酷炫的浏览器智能体界面，支持在线 3D 晶体分子查看。
*   **生成式晶体模型**：接入基于扩散模型的 CDVAE，取代当前的硬元素替换法，赋予模型真正的“空间无约束”造物能力。
*   **主动学习 (Active Learning)**：通过在大量生成的 AI_CIF 结构上循环验证，自动精调顶层的带隙和力场预测模型权重。

---

## 10. 致谢与参考文献
本系统的顺利构建离不开开源计算材料学社区的基石工作：
*   [Pymatgen](https://pymatgen.org/): Robust materials analysis.
*   [CHGNet](https://chgnet.lbl.gov/): Pretrained universal neural network potential (Deng et al., *Nature Machine Intelligence* 2023).
*   [MEGNet](https://github.com/materialsvirtuallab/megnet): Graph networks for materials (Chen et al., *Chemistry of Materials* 2019).
*   [Materials Project](https://materialsproject.org/): Open database for materials.
*   [LangChain](https://www.langchain.com/): Framework for building LLM applications.

**如果您在研究中使用了本系统的架构或理念，欢迎引用。**