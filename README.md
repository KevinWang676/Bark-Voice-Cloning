## Bark Voice Cloning + Multi‑Model TTS / Voice Cloning / Voice Conversion (UI + Notebooks)

**English (default)** | [**简体中文**](README_CN.md)

### Introduction
This repo **started as a single Bark voice cloning project** and has evolved into a collection of **cutting-edge TTS / voice cloning / voice conversion** training & inference scripts (UI + Colab notebooks).

It is a practical toolbox focused on:
- A ready-to-run **Gradio Web UI** for **Bark** voice cloning + TTS + voice conversion.
- A separate **Sambert UI** workflow for **Chinese (and bilingual) personal voice cloning** with data labeling → training → inference.
- A curated set of **Colab/Jupyter notebooks** covering multiple cutting-edge TTS / VC pipelines (GPT-SoVITS, XTTS, VALL-E X, F5‑TTS, CosyVoice, OpenAI TTS + VC, etc.).

## What's inside (Key entrypoints)
- **Bark Web UI**: `app.py`
  - Tabs: **Clone Voice** (create `.npz` prompt), **TTS**, **Voice Conversion**
  - Uses: `cloning/clonevoice.py`, `swap_voice.py`, `bark/`, `util/`, `training/`
- **Sambert Web UI**: `sambert-ui/app.py` (local), `sambert-ui/app_colab.py` (Colab-friendly)
- **Bark training utilities (experimental)**: `training/training_prepare.py`, `training/train.py`, `training/data.py`

## Quick Start (Bark UI)
### Requirements
- Python **3.10+** recommended
- GPU recommended (CPU works but is slow)

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
python app.py
```

### Downloads & outputs
- On first run, Bark checkpoints are downloaded into `./models/` (see `bark/generation.py`).
- HuBERT + tokenizer for voice cloning are downloaded into `./models/hubert/` (see `bark/hubert/hubert_manager.py`).
- Generated audio files are written to `outputs/` by default (configurable via `config.yaml` → `output_folder_path`).

### Important note for local runs
The Bark UI’s “Create Voice” feature writes a `.npz` prompt file. The default path in `app.py` is set for Colab (`/content/...`).  
If you run locally, you may need to update that destination path to a valid path on your machine (e.g. inside `bark/assets/prompts/`).

## Quick Start (Sambert UI)
Sambert UI provides a full pipeline: **auto labeling → training → inference**.

```bash
cd sambert-ui
pip install -r requirements.txt
python app.py
```

More details: `sambert-ui/README.md`

## Training & inference scripts (Bark path)
### Inference
- **TTS (text → audio)**:
  - Core API: `bark/api.py` (`generate_with_settings`, `semantic_to_waveform`)
  - UI wrapper: `app.py` (`generate_text_to_speech`)
- **Voice cloning (audio → .npz prompt)**:
  - `cloning/clonevoice.py` (HuBERT + tokenizer + EnCodec → save `.npz`)
- **Voice conversion (audio → new voice)**:
  - `swap_voice.py` (HuBERT tokens + Bark semantic_to_waveform with `history_prompt`)

### Training (experimental)
- `training/training_prepare.py`: generate semantic tokens from text, then synthesize wav pairs
- `training/train.py`: prepare HuBERT-ready features and trigger tokenizer training (calls `bark/hubert/customtokenizer.py`)
- `training/data.py`: text sourcing / filtering helpers

## Notebooks (Colab/Jupyter)
### Notebook organization
Voice-related notebooks are grouped under:
- `notebooks/tts/` (TTS / voice cloning)
- `notebooks/vc/` (voice conversion; **any notebook with `VC` in its filename**)

To keep older links working, the original notebook paths are kept as **symlinks**.

### TTS / Voice cloning notebooks
- **Bark**: [`Bark_Voice_Cloning.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/Bark_Voice_Cloning.ipynb), [`Bark_Coqui.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/Bark_Coqui.ipynb)
- **Sambert / Chinese voice cloning**: [`Voice_Cloning_for_Chinese_Speech_v2.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/Voice_Cloning_for_Chinese_Speech_v2.ipynb), [`SambertHifigan.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/SambertHifigan.ipynb), [`Sambert_Voice_Cloning_in_One_Click.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/Sambert_Voice_Cloning_in_One_Click.ipynb), [`Sambert_UI.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/sambert-ui/Sambert_UI.ipynb)
- **GPT-SoVITS**: [`GPT_SoVITS.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS.ipynb), [`GPT_SoVITS_2.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_2.ipynb), [`GPT_SoVITS_emo.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_emo.ipynb), [`GPT_SoVITS_v2_0808.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_v2_0808.ipynb), [`GPT_SoVITS_v3.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_v3.ipynb), [`GPT_SoVITS_v3_03_30.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_v3_03_30.ipynb), [`GPT_SoVITS_v4.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/GPT_SoVITS_v4.ipynb)
- **XTTS**: [`XTTS_Colab.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/XTTS_Colab.ipynb)
- **VALL‑E X**: [`VALL_E_X.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/VALL_E_X.ipynb)
- **F5‑TTS**: [`F5_TTS.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/F5_TTS.ipynb), [`F5_TTS_Training.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/F5_TTS_Training.ipynb)
- **CosyVoice**: [`CosyVoice.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/CosyVoice.ipynb), [`CosyVoice2.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/CosyVoice2.ipynb)
- **Other**: [`OpenVoice.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/OpenVoice.ipynb), [`Seamless_Meta.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/tts/Seamless_Meta.ipynb)

### Voice conversion (VC) notebooks
- **KNN‑VC**: [`KNN_VC.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/KNN_VC.ipynb)
- **NeuCoSVC**: [`NeuCoSVC.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/NeuCoSVC.ipynb), [`NeuCoSVC_v2_先享版.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/NeuCoSVC_v2_先享版.ipynb)
- **OpenAI TTS + VC**: [`OpenAI_TTS_KNN_VC.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/OpenAI_TTS_KNN_VC.ipynb), [`OpenAI_TTS_KNN_VC_en.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/OpenAI_TTS_KNN_VC_en.ipynb), [`OpenAI_TTS_RVC.ipynb`](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/vc/OpenAI_TTS_RVC.ipynb)

## Repo layout
```text
.
├── app.py                      # Bark Gradio UI (voice cloning / TTS / voice conversion)
├── bark/                        # Bark core + HuBERT utilities
├── cloning/                     # Voice cloning (audio -> .npz prompt)
├── training/                    # Experimental training utilities
├── swap_voice.py                # Voice conversion helper
├── util/                        # Settings + SSML/text helpers
├── config.yaml                  # UI + output configuration
├── sambert-ui/                  # Sambert UI (label/train/infer)
└── notebooks/
    ├── tts/                     # TTS / voice cloning notebooks
    ├── vc/                      # Voice conversion notebooks (filenames contain "VC")
    └── ...                      # Other notebooks (LLM/agent/video/etc.)
```

## Disclaimer
This repository is intended for research and learning. Please comply with local laws and obtain proper consent before cloning or converting any voice.

## Original README
### Bark Voice Cloning 🐶 & Voice Cloning for Chinese Speech 🎶
### [简体中文](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/README_zh.md)
## 1️⃣ Bark Voice Cloning

> 10/19/2023: Fixed `ERROR: Exception in ASGI application` by specifying `gradio==3.33.0` and `gradio_client==0.2.7` in [requirements.txt](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/requirements.txt).

> 11/08/2023: Integrated [KNN-VC](https://github.com/bshall/knn-vc) into [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech) and created an easy-to-use Gradio interface. Try it [here](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/OpenAI_TTS_KNN_VC_en.ipynb).

> 02/27/2024: We are thrilled to launch our most powerful **AI song cover generator** ever with [Shanghai Artificial Intelligence Laboratory](https://www.shlab.org.cn/)! Just need to provide the name of a song and our application running on an **A100** GPU will handle everything else. Check it out in our [**website**](https://www.talktalkai.com/) (please click "EN" in the first tab of our website to see the english version)! 💕

Based on [bark-gui](https://github.com/C0untFloyd/bark-gui) and [bark](https://github.com/suno-ai/bark). Thanks to [C0untFloyd](https://github.com/C0untFloyd).

**Quick start**: [**Colab Notebook**](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Bark_Voice_Cloning.ipynb) ⚡

**HuggingFace Demo**: [**Bark Voice Cloning**](https://huggingface.co/spaces/kevinwang676/Bark-with-Voice-Cloning) 🤗 (Need a GPU)

**Demo Video**: [**YouTube Video**](https://www.youtube.com/watch?v=IAf695dhkUc&t=4s)

If you would like to run the code locally, remember to replace the original path `/content/Bark-Voice-Cloning/bark/assets/prompts/file.npz` with the path of `file.npz` in your own computer.

### If you like the quick start, please star this repository. ⭐⭐⭐

## Easy to use: 

(1) First upload audio for voice cloning and click `Create Voice`.

![image](https://github.com/KevinWang676/Bark-Voice-Cloning/assets/126712357/65e2b695-f529-4fb5-9549-4e86e6a4d8b2)

(2) Choose the option called "file" in `Voice` if you'd like to use voice cloning.

(3) Click `Generate`. Done!

![image](https://github.com/KevinWang676/Bark-Voice-Cloning/assets/126712357/20911e37-768d-47d5-bb86-d12a3ab04c5d)

## 2️⃣ Voice Cloning for Chinese Speech
> 10/26/2023: Integrated labeling, training and inference into an easy-to-use user interface of SambertHifigan. Thanks to [wujohns](https://github.com/wujohns).

We want to point out that [Bark](https://github.com/suno-ai/bark) is very good at generating English speech but relatively poor at generating Chinese speech. So we'd like to adopt another approach, which is called [SambertHifigan](https://www.modelscope.cn/models/speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_16k/summary), to realizing voice cloning for Chinese speech. Please check out our [Colab Notebook](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Voice_Cloning_for_Chinese_Speech_v2.ipynb) for the implementation.

Quick start: [Colab Notebook](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/sambert-ui/Sambert_UI.ipynb) ⚡

HuggingFace demo: [Voice Cloning for Chinese Speech](https://huggingface.co/spaces/kevinwang676/Personal-TTS) 🤗

[![Star History Chart](https://api.star-history.com/svg?repos=KevinWang676/Bark-Voice-Cloning&type=Date)](https://star-history.com/#KevinWang676/Bark-Voice-Cloning&Date)
