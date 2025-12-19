## Bark 声音克隆 + 多模型 TTS / 声音克隆 / 变声（UI + Colab 笔记本）

[**English**](README.md) | **简体中文**

### 简介
本仓库最初是一个 **Bark 声音克隆** 项目，现已发展为一个聚合多种前沿 **TTS / 声音克隆 / 变声（VC）** 的训练与推理脚本合集（含 UI 与 Colab 笔记本）。

目前主要包含：
- 基于 **Bark** 的 **Gradio 可视化 Web UI**（声音克隆 + TTS + 变声）。
- 面向中文（并支持中英双语）的 **Sambert UI**：数据标注 → 训练 → 推理一体化界面。
- 大量可直接运行的 **Colab/Jupyter 笔记本**，覆盖多种前沿 TTS/VC 技术路线（GPT‑SoVITS、XTTS、VALL‑E X、F5‑TTS、CosyVoice、OpenAI TTS + 变声等）。

## 主要入口
- **Bark Web UI**: `app.py`
  - 标签页：**Clone Voice**（生成 `.npz` 提示音）、**TTS**、**Voice Conversion**（变声）
  - 调用：`cloning/clonevoice.py`、`swap_voice.py`、`bark/`、`util/`、`training/`
- **Sambert Web UI**: `sambert-ui/app.py`（本地）、`sambert-ui/app_colab.py`（Colab 友好）
- **Bark 训练工具（实验性质）**: `training/training_prepare.py`、`training/train.py`、`training/data.py`

## 快速开始（Bark UI）
### 环境要求
- 推荐 Python **3.10+**
- 推荐 GPU（CPU 可用但速度较慢）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动
```bash
python app.py
```

### 模型下载与输出目录
- 首次运行时，Bark 检查点会下载到 `./models/`（见 `bark/generation.py`）。
- 用于声音克隆的 HuBERT + tokenizer 会下载到 `./models/hubert/`（见 `bark/hubert/hubert_manager.py`）。
- 生成的音频文件默认写入 `outputs/`（可通过 `config.yaml` → `output_folder_path` 配置）。

### 本地运行重要说明
Bark UI 的"Create Voice"功能会写入 `.npz` 提示音文件。`app.py` 中的默认路径是为 Colab 设置的（`/content/...`）。  
如果在本地运行，您可能需要将该目标路径更新为您机器上的有效路径（例如 `bark/assets/prompts/` 内）。

## 快速开始（Sambert UI）
Sambert UI 提供完整流程：**自动标注 → 训练 → 推理**。

```bash
cd sambert-ui
pip install -r requirements.txt
python app.py
```

更多详情：`sambert-ui/README.md`

## 训练与推理脚本（Bark 路线）
### 推理
- **TTS（文本 → 音频）**：
  - 核心 API：`bark/api.py`（`generate_with_settings`、`semantic_to_waveform`）
  - UI 封装：`app.py`（`generate_text_to_speech`）
- **声音克隆（音频 → .npz 提示音）**：
  - `cloning/clonevoice.py`（HuBERT + tokenizer + EnCodec → 保存 `.npz`）
- **变声（音频 → 新声音）**：
  - `swap_voice.py`（HuBERT tokens + Bark semantic_to_waveform + `history_prompt`）

### 训练（实验性质）
- `training/training_prepare.py`：从文本生成语义 token，然后合成 wav 对
- `training/train.py`：准备 HuBERT 就绪的特征并触发 tokenizer 训练（调用 `bark/hubert/customtokenizer.py`）
- `training/data.py`：文本来源 / 过滤辅助工具

## 笔记本（Colab/Jupyter）
### 笔记本组织方式
语音相关笔记本按以下方式分组：
- `notebooks/tts/`（TTS / 声音克隆）
- `notebooks/vc/`（变声；**文件名包含 `VC` 的所有笔记本**）

### TTS / 声音克隆笔记本
- **Bark**：[`Bark_Voice_Cloning.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/Bark_Voice_Cloning.ipynb)、[`Bark_Coqui.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/Bark_Coqui.ipynb)
- **Sambert / 中文声音克隆**：[`Voice_Cloning_for_Chinese_Speech_v2.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/Voice_Cloning_for_Chinese_Speech_v2.ipynb)、[`SambertHifigan.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/SambertHifigan.ipynb)、[`Sambert_Voice_Cloning_in_One_Click.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/Sambert_Voice_Cloning_in_One_Click.ipynb)、[`Sambert_UI.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/sambert-ui/Sambert_UI.ipynb)
- **GPT-SoVITS**：[`GPT_SoVITS.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS.ipynb)、[`GPT_SoVITS_2.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_2.ipynb)、[`GPT_SoVITS_emo.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_emo.ipynb)、[`GPT_SoVITS_v2_0808.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_v2_0808.ipynb)、[`GPT_SoVITS_v3.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_v3.ipynb)、[`GPT_SoVITS_v3_03_30.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_v3_03_30.ipynb)、[`GPT_SoVITS_v4.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_v4.ipynb)
- **XTTS**：[`XTTS_Colab.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/XTTS_Colab.ipynb)
- **VALL‑E X**：[`VALL_E_X.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/VALL_E_X.ipynb)
- **F5‑TTS**：[`F5_TTS.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/F5_TTS.ipynb)、[`F5_TTS_Training.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/F5_TTS_Training.ipynb)
- **CosyVoice**：[`CosyVoice.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/CosyVoice.ipynb)、[`CosyVoice2.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/CosyVoice2.ipynb)
- **其他**：[`OpenVoice.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/OpenVoice.ipynb)、[`Seamless_Meta.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/Seamless_Meta.ipynb)

### 变声（VC）笔记本
- **KNN‑VC**：[`KNN_VC.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/KNN_VC.ipynb)
- **NeuCoSVC**：[`NeuCoSVC.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/NeuCoSVC.ipynb)、[`NeuCoSVC_v2_先享版.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/NeuCoSVC_v2_先享版.ipynb)
- **OpenAI TTS + VC**：[`OpenAI_TTS_KNN_VC.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/OpenAI_TTS_KNN_VC.ipynb)、[`OpenAI_TTS_KNN_VC_en.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/OpenAI_TTS_KNN_VC_en.ipynb)、[`OpenAI_TTS_RVC.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/OpenAI_TTS_RVC.ipynb)

## 目录结构
```text
.
├── app.py                      # Bark Gradio UI（声音克隆 / TTS / 变声）
├── bark/                        # Bark 核心 + HuBERT 工具
├── cloning/                     # 声音克隆（音频 -> .npz 提示音）
├── training/                    # 实验性训练工具
├── swap_voice.py                # 变声辅助工具
├── util/                        # 设置 + SSML/文本辅助工具
├── config.yaml                  # UI + 输出配置
├── sambert-ui/                  # Sambert UI（标注/训练/推理）
└── notebooks/
    ├── tts/                     # TTS / 声音克隆笔记本
    ├── vc/                      # 变声笔记本（文件名包含"VC"）
    └── ...                      # 其他笔记本（LLM/agent/video 等）
```

## 免责声明
本仓库仅供研究和学习使用。请遵守当地法律，在克隆或转换任何声音之前需要取得音色提供者的明确同意。

[![Star History Chart](https://api.star-history.com/svg?repos=KevinWang676/Bark-Voice-Cloning&type=Date)](https://star-history.com/#KevinWang676/Bark-Voice-Cloning&Date)
