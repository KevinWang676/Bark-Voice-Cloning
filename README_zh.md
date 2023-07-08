# 第二代Bark声音克隆 🐶 & 全新中文声音克隆 🎶

## 1️⃣ 第二代Bark声音克隆

在线快速运行：[Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Bark_Voice_Cloning.ipynb) ⚡

HuggingFace在线程序：[Bark声音克隆](https://huggingface.co/spaces/kevinwang676/Bark-with-Voice-Cloning) 🤗

使用指南：[B站视频](https://www.bilibili.com/video/BV16g4y1N7ZG) 📺

_注：运行时需要使用GPU_

#### 如果您喜欢这个项目，请在Github上点赞吧！ ⭐⭐⭐

## 2️⃣ 全新中文声音克隆

### 训练5分钟，通话不限时！ 🌞

因为[Bark](https://github.com/suno-ai/bark)中文文本转语音的功能远远不如英文的效果好，所以我们采用一种新的技术路径[SambertHifigan](https://www.modelscope.cn/models/speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_16k/summary)来实现中文的声音克隆功能。

### 如何使用 💡

#### (1) 准备并上传一段中文语音：单一说话人、长度一分钟左右的`.wav`文件。

我们的程序能够自动将您上传的语音切片。您可以使用我们制作的[专属工具](https://kevinwang676-test-1.hf.space/)从B站直接提取视频中的语音，只需要填写视频的BV号和起止时间。

#### (2) 使用我们的[Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Voice_Cloning_for_Chinese_Speech.ipynb)运行程序：运行所有代码即可。

您可以在Colab笔记本的`推理`代码模块更改中文文本，进而输出您想要的内容。运行笔记本时的**注意事项**：

* 需要在运行完所有`pip install`命令后，点击Colab左下角终端，依次执行
```
apt-get install sox

cd pytorch_wavelets

pip install .
```
* 上传音频素材后，需要将代码`split_long_audio(whisper_model, "filename.wav", "test", "dataset_raw")`中的`filename`替换成音频文件的名字
* 需要在Colab中新建三个文件夹，分别名为：`test_wavs`，`output_training_data`，`pretrain_work_dir`
* 训练完成后，在推理模块的`output = inference(input="大家好呀，欢迎使用滔滔智能的声音克隆产品！")`代码处可以自由编辑中文文本，实现中文声音克隆
* 整个过程都需要使用GPU

### 三种使用方式 😄

在线快速运行: [Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Voice_Cloning_for_Chinese_Speech.ipynb) ⚡

HuggingFace在线程序: [全新中文声音克隆](https://huggingface.co/spaces/kevinwang676/Personal-TTS) 🤗

阿里云笔记本在线运行：您也可以免费使用阿里云提供的[笔记本](https://modelscope.cn/models/damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/summary)进行训练。进入页面后点击右上角的`Notebook快速开发`，选择GPU环境，上传[Colab笔记本（可下载）](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/Voice_Cloning_for_Chinese_Speech.ipynb)和`.wav`文件素材后就能够以同样的方式运行啦！ 🍻
