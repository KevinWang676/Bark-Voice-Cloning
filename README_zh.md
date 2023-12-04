# 第二代Bark声音克隆 🐶 & 全新中文声音克隆 🎶

## 1️⃣ 第二代Bark声音克隆

> 11/08/2023更新：将AI变声模型[KNN-VC](https://github.com/bshall/knn-vc)与最新发布的[OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech)结合，实现更加真实的AI变声，您可以[在线体验](https://huggingface.co/spaces/kevinwang676/OpenAI-TTS-Voice-Conversion)或在[Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/OpenAI_TTS_KNN_VC.ipynb)中运行

> 11/13/2023更新：将声音转换模型[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)与最新发布的[OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech)结合，您可以使用[Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/OpenAI_TTS_RVC.ipynb)运行，点击[这里](https://github.com/KevinWang676/Bark-Voice-Cloning/assets/126712357/e7fa4d21-d616-41b1-be34-5d420f65c943)试听效果

> 11/23/2023更新：Sambert声音克隆在线体验的bug已修复，[点击使用](https://huggingface.co/spaces/kevinwang676/Personal-TTS)

> 12/01/2023更新：ChatGLM2神里绫华模型+Bert-VITS2文本转语音，和绫华一起谈天说地吧，点击[Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/ChatGLM2_linghua_VITS2.ipynb)运行，[在线使用](https://kevinwang676-chatglm2-bert-vits2-lh.hf.space)

> 12//03/2023更新：Sambert声音克隆本地部署教程以及[Sambert UI Colab](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/sambert-ui/Sambert_UI.ipynb)已上传，[点击查看](https://github.com/KevinWang676/Bark-Voice-Cloning/tree/main/sambert-ui)

在线快速运行：[Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Bark_Voice_Cloning.ipynb) ⚡

HuggingFace在线程序：[Bark声音克隆](https://huggingface.co/spaces/kevinwang676/Bark-with-Voice-Cloning) 🤗

使用指南：[B站视频](https://www.bilibili.com/video/BV16g4y1N7ZG) 📺

_注：(1) Bark声音克隆功能基于[bark-gui](https://github.com/C0untFloyd/bark-gui)项目；(2) 运行时需要使用GPU_

#### 如果您喜欢这个项目，请在Github上点赞吧！ ⭐⭐⭐

## 2️⃣ VALL-E X 全新声音克隆
> 08/26/2023更新：VALL-E X 声音克隆，支持中日英三语；只需3秒语音，即可快速复刻您喜欢的音色

[VALL-E X](https://www.microsoft.com/en-us/research/project/vall-e-x/)是由微软团队开发的支持多语种的语音合成模型，此部分基于Plachtaa的开源项目[VALL-E-X](https://github.com/Plachtaa/VALL-E-X)，进行了用户界面和功能上的优化。您可以使用我们制作的[专属工具](https://kevinwang676-voicechangers.hf.space/)从B站直接提取视频中的语音，只需要填写视频的BV号和起止时间。

Colab快速启动: [Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/VALL_E_X.ipynb)

HuggingFace在线程序: [VALL-E X在线](https://huggingface.co/spaces/kevinwang676/VALLE) 🤗

## 3️⃣ SambertHifigan中文声音克隆
> 07/19/2023更新：在执行`pip install kantts -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html`前需要先执行`pip install pysptk --no-build-isolation` (已在对应的Colab笔记本中更新)

> 08/27/2023更新：已修复SambertHifigan对应的Colab Notebook中的所有bug，[点击此处使用](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Voice_Cloning_for_Chinese_Speech_v2.ipynb)

> 09/09/2023更新：增加SambertHifigan中文声音克隆的在线一键启动版 [Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Sambert_Voice_Cloning_in_One_Click.ipynb)，使用更加方便 🌟

> 09/12/2023更新：增加[AutoDL镜像](https://www.codewithgpu.com/i/KevinWang676/Bark-Voice-Cloning/Sambert-VC)，支持在线GPU一键部署，快速开启声音克隆之旅 🍻

### 训练5分钟，通话不限时！ 🌞

因为[Bark](https://github.com/suno-ai/bark)文本转语音的中文效果远远不如英文的效果好，所以我们采用一种新的技术路径[SambertHifigan](https://www.modelscope.cn/models/speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_16k/summary)来实现中文的声音克隆功能。

### 如何使用 💡 [视频教程](https://www.bilibili.com/video/BV1Ch4y1Z7K6)

### (1) 准备并上传一段中文语音：单一说话人、长度一分钟左右的`.wav`文件。

我们的程序能够自动将您上传的语音切片。您可以使用我们制作的[专属工具](https://kevinwang676-voicechangers.hf.space/)从B站直接提取视频中的语音，只需要填写视频的BV号和起止时间。为了达到更好的声音克隆效果，中文语音素材需要符合以下**要求**：

* 音频尽量是干净人声，不要有BGM，不要有比较大的杂音，不要有一些特殊的声效，比如回声等
* 声音的情绪尽量稳定，以说话的语料为主，不要是『嗯』『啊』『哈』之类的语气词

### (2) 使用我们的[Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Voice_Cloning_for_Chinese_Speech_v2.ipynb)运行程序：运行所有代码即可。

您可以在Colab笔记本的`推理`代码模块更改中文文本，进而输出您想要的内容。运行笔记本时的**注意事项**：

* 上传音频素材后，需要将代码`split_long_audio(whisper_model, "filename.wav", "test", "dataset_raw")`中的`filename`替换成音频文件的名字
* 需要在Colab中新建三个文件夹，分别名为：`test_wavs`，`output_training_data`，`pretrain_work_dir`
* 训练完成后，在推理模块的`output = inference(input="大家好呀，欢迎使用滔滔智能的声音克隆产品！")`代码处可以自由编辑中文文本，实现中文声音克隆
* 整个过程都需要使用GPU；如果使用阿里云笔记本，则不需要在终端中执行第一步的环境设置

### (3) 一键在HuggingFace上免费部署 🤗

* 完成训练后，在Colab或阿里云笔记本中运行`!zip -r ./model.zip ./pretrain_work_dir`打包模型文件夹，下载并解压到本地
* 点击进入[HuggingFace程序](https://huggingface.co/spaces/kevinwang676/Personal-TTS)，点击右上角的三个圆点，选择`Duplicate this Space`将程序复制到自己的HuggingFace主页
* 点击进入`Files`，选择右上角`Add file`后，点击`Upload files`，将解压后的文件夹`pretrain_work_dir`从本地直接拖拽上传；需要先删除原有的`pretrain_work_dir`同名文件夹

### 四种使用方式 😄

**推荐**🌟 阿里云笔记本在线运行：您也可以**免费**使用阿里云提供的[在线笔记本](https://modelscope.cn/models/damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/summary)进行训练，进入页面后点击右上角的`Notebook快速开发`，选择GPU环境，上传代码文件[阿里云专属笔记本（可下载）](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/%E9%98%BF%E9%87%8C%E4%BA%91%E7%AC%94%E8%AE%B0%E6%9C%AC%E8%AE%AD%E7%BB%83.ipynb)和`.wav`文件素材后就能够以同样的方式运行啦！ 🍻

Colab在线快速运行: [Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Voice_Cloning_for_Chinese_Speech_v2.ipynb) ⚡

HuggingFace在线程序: [全新中文声音克隆](https://huggingface.co/spaces/kevinwang676/Personal-TTS) 🤗

阿里魔搭社区在线程序：[个人声音定制](https://modelscope.cn/studios/damo/personal_tts/summary) 🎤

### 一键运行版本：最新[Colab笔记本](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Sambert_Voice_Cloning_in_One_Click.ipynb)（有时不稳定）及[AutoDL镜像](https://www.codewithgpu.com/i/KevinWang676/Bark-Voice-Cloning/Sambert-VC)运行（推荐）

* 最简洁的操作界面，一键上传语音素材，无需修改代码
* 点击进入[AutoDL镜像](https://www.codewithgpu.com/i/KevinWang676/Bark-Voice-Cloning/Sambert-VC)，创建新实例，按照操作指南即可快速开启声音克隆之旅 🎶

**注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用，严禁商业化运作。用户上传的语音及生成的内容均与本代码仓库所有者无关。**

![image](https://github.com/KevinWang676/Bark-Voice-Cloning/assets/126712357/7597122b-307f-41de-abdd-454dc0db5271)

[![Star History Chart](https://api.star-history.com/svg?repos=KevinWang676/Bark-Voice-Cloning&type=Date)](https://star-history.com/#KevinWang676/Bark-Voice-Cloning&Date)
