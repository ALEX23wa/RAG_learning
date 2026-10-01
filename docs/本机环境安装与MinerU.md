# 本机环境与 MinerU

## 安装

本机 Conda 位于 `E:\NVIDIA`。环境按当前 Conda 默认路径安装在 `C:\Users\23245\.conda\envs\langchain`；包缓存遵循 `.condarc`，优先使用 `C:\Users\23245\.conda\pkgs`。

在项目根目录的 PowerShell 中运行：

```powershell
.\scripts\install-environment.ps1
```

脚本使用 `conda env create/update -f environment.yml`，随后执行 `pip check` 和 `visual_bge` 导入验证。默认通过环境名 `langchain` 遵循 Conda 的环境与缓存配置，不再强制使用 D 盘；需要自定义时可以传入 `-CondaExe`、`-EnvironmentPath`、`-CacheRoot`。

如需在当前 PowerShell 激活环境：

```powershell
(& 'E:\NVIDIA\Scripts\conda.exe' 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate langchain
```

也可以直接使用 `C:\Users\23245\.conda\envs\langchain\python.exe`，不依赖 PATH。

原环境文件有两个无法从 PyPI 安装的包，已修复来源：

- `en-core-web-sm`：使用 spaCy 官方 3.8.0 wheel。
- `visual-bge`：使用 FlagEmbedding 官方仓库的固定提交和 `research/visual_bge` 子目录，按上游说明可编辑安装，并启用 setuptools 的 `editable_mode=compat`。普通 wheel 和默认可编辑模式会漏掉命名空间包，导致安装成功但无法导入。安装脚本将源码保存在 `C:\Users\23245\.conda\src`，使用环境期间不要删除该目录。

移除了另一台电脑的 `prefix: D:\Anaconda\envs\langchain`。保留原有依赖版本；此文件仍是 Windows 环境定义，并非跨平台环境。

## 模型和缓存

`data_process/data_md.py` 在导入 MinerU 前设置默认路径，允许通过环境变量覆盖：

| 内容 | 默认路径 |
| --- | --- |
| MinerU 配置与数据根目录 | `D:\Program Files\MinerU` |
| 模型权重 | `D:\Program Files\MinerU\models` |
| Hugging Face 缓存 | `D:\Program Files\MinerU\cache\huggingface` |
| ModelScope 缓存 | `D:\Program Files\MinerU\cache\modelscope` |

模型根目录由 `MINERU_MODEL_BASE_DIR` 控制，`MINERU_HOME` 不是 Python 包的安装目录。Python 包安装在 Conda 环境中。自定义路径需要在导入 `data_md` 或 MinerU 前设置；已有 `MINERU_CONFIG` 或缓存变量也可能影响配置。

在独立命令行下载/校验模型时，同样显式设置这些路径：

```powershell
$env:MINERU_HOME = 'D:\Program Files\MinerU'
$env:MINERU_MODEL_BASE_DIR = "$env:MINERU_HOME\models"
$env:HF_HOME = "$env:MINERU_HOME\cache\huggingface"
$env:HF_HUB_CACHE = "$env:HF_HOME\hub"
$env:HF_XET_CACHE = "$env:HF_HOME\xet"
$env:MODELSCOPE_CACHE = "$env:MINERU_HOME\cache\modelscope"
mineru-kit models download --tier standard --small-backend onnx --vlm-engine llama-cpp --source modelscope
mineru-kit models verify --tier standard --small-backend onnx --vlm-engine llama-cpp
```

## GPU 优先与 CPU 回退

本机显卡是 **AMD Radeon RX 6650 XT，8176 MiB 显存**。`mineru-llama-cpp==0.1.2` 的 Windows wheel 自带 Vulkan 后端，设备枚举已识别 `Vulkan0`。无需 NVIDIA CUDA；`torch.cuda.is_available()` 不能判断 AMD Vulkan 能否使用。

本地解析采用 `onnx + llama-cpp`。MinerU 4.0.9 的 ONNX 小模型固定在 CPU 上运行，VLM 使用 Vulkan GPU，所以不能把整个 PDF 流程称为全部在 GPU 上运行。`flash`/`basic` 不一定调用 VLM；验证 GPU 应使用 `standard` 或 `advanced`。

```powershell
python data_process/data_md.py papers/rag.pdf --page-range 1 --device auto
python data_process/data_md.py papers/rag.pdf --page-range 1 --device gpu
python data_process/data_md.py papers/rag.pdf --page-range 1 --device cpu
```

- `auto`：用实际安装的 `llama-server --list-devices` 探测，优先 GPU；无可用设备时用 CPU。GPU 内存/设备错误或 native 崩溃时，在新的 CPU 进程中重试一次。
- `gpu`：强制 GPU，失败直接报告，适合诊断。
- `cpu`：跳过 GPU 探测，隐藏 Vulkan 设备，强制 CPU。

解析错误、模型下载失败等不会被悄悄当作 GPU 故障重试。CPU 重试失败会保留并报告错误。输出目录保留 `mineru-gpu.log`、`mineru-cpu.log`；原来的默认目录 `papers/extracted/mineru` 和 `markdown.md` 文件名保持不变，多份 PDF 应传入不同 `--output-dir`。

已有 `GGML_VK_VISIBLE_DEVICES` 限制会被 GPU 探测尊重；若此前手动设为空格，会禁用 GPU，需要在当前终端删除该变量：`Remove-Item Env:GGML_VK_VISIBLE_DEVICES`。

## 本机验证记录（2026-09-30）

- 环境已从 `D:\conda\envs\langchain` 克隆至 Conda 默认目录 `C:\Users\23245\.conda\envs\langchain`；264 个已安装包的版本一致。已重新生成命令行入口，将 Visualized-BGE 源码迁移到 `C:\Users\23245\.conda\src`。新环境通过依赖检查、8 项测试和 GPU PDF 实测，结果保存在 `papers/extracted/verify-migrated`。
- Windows x64，Intel i5-12400F，AMD Radeon RX 6650 XT；Python 3.12.14。
- 最终环境已通过 `conda env update -f environment.yml` 和 `pip check`；259 条 pip 安装要求已核对，固定版本无偏差。LangChain、FlagEmbedding、Visualized-BGE、PyTorch、Paddle、spaCy 与 unstructured 等关键模块可导入，spaCy 英文模型可加载。
- `mineru-kit models verify`：ONNX 与 GGUF 两套权重均正常。
- GPU 原生日志：`offloaded 25/25 layers to GPU`，`CLIP using Vulkan0 backend`。
- `rag.pdf` 第 1 页，`standard`：GPU 与 CPU 均成功输出 Markdown；最终环境自动选择 GPU 的模型推理时间约 9.7 秒；隐藏 Vulkan 设备后，`auto` 自动选择 CPU 并成功完成，约 22.8 秒。这是单页功能验证，未包含完整下载/启动耗时，不代表所有 PDF 的速度。
- 输出及日志保存在 `papers/extracted/verify-auto`、`papers/extracted/verify-fallback`；较早的 GPU、CPU 与纯文本验证分别保存在 `verify-gpu`、`verify-cpu`、`verify-flash`，GPU 加载证据见 `verify-gpu/gpu-offload.log`。这些生成文件不纳入 Git。
- `python -m unittest discover -s tests -v`：覆盖 GPU 成功、无设备、显存错误回退、CPU 失败、下载错误不重试、强制设备和 native 崩溃分类，共 8 项通过。回退故障使用模拟测试，没有人为破坏显卡驱动。

## 参考资料

- [MinerU 模型路径与下载配置](https://opendatalab.github.io/MinerU/zh/usage/model_source/)
- [Visualized-BGE 官方安装说明](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/visual_bge/README.md)
