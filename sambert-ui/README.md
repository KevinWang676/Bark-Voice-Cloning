# Sambert UI 使用指南 📒
### [Colab](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/sambert-ui/Sambert_UI.ipynb) 在线使用 Sambert UI      [视频教程](https://www.bilibili.com/video/BV1AN411j7zV/?spm_id_from=333.999.0.0)
## 1. 环境配置

```
git clone https://github.com/KevinWang676/Bark-Voice-Cloning
cd Bark-Voice-Cloning
cd sambert-ui
pip install -r requirements.txt
sudo apt install build-essential
pip install kantts -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install tts-autolabel -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
apt-get install sox # 也可以选择 pip install sox 来安装sox依赖
```

安装PyTorch环境（若已安装PyTorch，可跳过此步）
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchtext==0.14.1 torchaudio==0.13.1 torchdata==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## 2. 加载 Sambert UI
```
python app.py
```

开启 Sambert 中英声音克隆之旅吧 💕

![image](https://github.com/KevinWang676/Bark-Voice-Cloning/assets/126712357/5b97ee5f-2595-46d9-97d2-d41984c583f5)
