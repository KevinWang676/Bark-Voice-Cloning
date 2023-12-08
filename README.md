# Bark Voice Cloning 🐶 & Voice Cloning for Chinese Speech 🎶
### [简体中文](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/README_zh.md)
## 1️⃣ Bark Voice Cloning

> 10/19/2023: Fixed `ERROR: Exception in ASGI application` by specifying `gradio==3.33.0` and `gradio_client==0.2.7` in [requirements.txt](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/requirements.txt).

> 11/08/2023：Integrated [KNN-VC](https://github.com/bshall/knn-vc) into [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech) and created an easy-to-use Gradio interface. Try it [here](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/OpenAI_TTS_KNN_VC_en.ipynb).

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

Quick start: [Colab Notebook](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Voice_Cloning_for_Chinese_Speech_v2.ipynb) ⚡

HuggingFace demo: [Voice Cloning for Chinese Speech](https://huggingface.co/spaces/kevinwang676/Personal-TTS) 🤗

[![Star History Chart](https://api.star-history.com/svg?repos=KevinWang676/Bark-Voice-Cloning&type=Date)](https://star-history.com/#KevinWang676/Bark-Voice-Cloning&Date)
