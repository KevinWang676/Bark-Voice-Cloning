## Bark Voice Cloning + Multi‑Model TTS / Voice Cloning / Voice Conversion (UI + Notebooks)

### Language
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
- **Bark**: `notebooks/tts/Bark_Voice_Cloning.ipynb`, `notebooks/tts/Bark_Coqui.ipynb`
- **Sambert / Chinese voice cloning**: `notebooks/tts/Voice_Cloning_for_Chinese_Speech_v2.ipynb`, `notebooks/tts/SambertHifigan.ipynb`, `notebooks/tts/Sambert_Voice_Cloning_in_One_Click.ipynb`, `sambert-ui/Sambert_UI.ipynb`
- **GPT-SoVITS**: `notebooks/tts/GPT_SoVITS.ipynb` (+ variants)
- **XTTS**: `notebooks/tts/XTTS_Colab.ipynb`
- **VALL‑E X**: `notebooks/tts/VALL_E_X.ipynb`
- **F5‑TTS**: `notebooks/tts/F5_TTS.ipynb`, `notebooks/tts/F5_TTS_Training.ipynb`
- **CosyVoice**: `notebooks/tts/CosyVoice.ipynb`, `notebooks/tts/CosyVoice2.ipynb`
- **Other**: `notebooks/tts/OpenVoice.ipynb`, `notebooks/tts/Seamless_Meta.ipynb`

### Voice conversion (VC) notebooks
- **KNN‑VC**: `notebooks/vc/KNN_VC.ipynb`
- **NeuCoSVC**: `notebooks/vc/NeuCoSVC*.ipynb`
- **OpenAI TTS + VC**: `notebooks/vc/OpenAI_TTS_KNN_VC*.ipynb`, `notebooks/vc/OpenAI_TTS_RVC.ipynb`

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
