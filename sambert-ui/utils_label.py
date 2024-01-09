# 对 sambert 训练的数据标注处理
import os
import shutil
import uuid
import librosa
import gradio as gr
from scipy.io import wavfile
import numpy as np
import whisper
from modelscope.tools import run_auto_label
from utils_base import ensure_empty_dir, datasets_dir, get_dataset_list

# 绝对路径获取方法
curPath = os.path.dirname(os.path.abspath(__file__))
def getAbsPath (relativePath):
  joinPath = os.path.join(curPath, relativePath)
  return os.path.normpath(
    os.path.abspath(joinPath)
  )

# 初始化 whisper 模型的加载
model_path = getAbsPath('../../models/whisper/medium.pt')
whisper_model = None
if shutil.os.path.exists(model_path):
  whisper_model = whisper.load_model(model_path)
else:
  whisper_model = whisper.load_model('medium')

# whisper 音频分割方法 ----------------------------------------------
def split_long_audio(model, filepaths, save_path, out_sr=44100):
  # 格式化输入的音频路径(兼容单个音频和多个音频)
  if isinstance(filepaths, str):
    filepaths = [filepaths]

  # 对音频依次做拆分并存放到临时路径
  for file_idx, filepath in enumerate(filepaths):
    print(f"Transcribing file {file_idx}: '{filepath}' to segments...")
    result = model.transcribe(filepath, word_timestamps=True, task="transcribe", beam_size=5, best_of=5)
    segments = result['segments']

    # 采用 librosa 配合 scipy 做音频数据分割
    wav, sr = librosa.load(filepath, sr=None, offset=0, duration=None, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    peak = np.abs(wav).max()
    if peak > 1.0:
      wav = 0.98 * wav / peak
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=out_sr)
    wav2 /= max(wav2.max(), -wav2.min())

    # 将长音频文件分割成一条条的短音频并放入指定的目录
    for i, seg in enumerate(segments):
      start_time = seg['start']
      end_time = seg['end']
      wav_seg = wav2[int(start_time * out_sr):int(end_time * out_sr)]
      wav_seg_name = f"{file_idx}_{i}.wav"
      out_fpath = os.path.join(save_path, wav_seg_name)
      wavfile.write(out_fpath, rate=out_sr, data=(wav_seg * np.iinfo(np.int16).max).astype(np.int16))

# 自动标注与标注后的文件打包 --------------------------------------------
def auto_label(audio, name):
  if not audio or not name:
    return '', gr.update(choices=get_dataset_list())

  # 创建临时目录用于存放分割后的音频与再次标注的信息
  input_wav = getAbsPath(f'./temp/input-{ uuid.uuid4() }')
  ensure_empty_dir(input_wav)

  work_dir = os.path.join(datasets_dir, name)
  ensure_empty_dir(work_dir)

  # 音频分割
  split_long_audio(whisper_model, audio, input_wav)

  # 音频自动标注
  # 第一次会自动下载对应的模型
  run_auto_label(
    input_wav=input_wav,
    work_dir=work_dir,
    resource_revision='v1.0.7'
  )

  # 移除目录
  shutil.rmtree(input_wav)

  # 返回结果
  return '标注成功', gr.update(choices=get_dataset_list())
  
# 删除数据集 ----------------------------------------------------
# name - 删除的数据集名称
def delete_dataset(name):
  try:
    if not name:
      return gr.update(choices=get_dataset_list())

    target_dir = os.path.join(datasets_dir, name)
    shutil.rmtree(target_dir)
    return gr.update(choices=get_dataset_list(), value=None)
  except Exception:
    return gr.update(choices=get_dataset_list(), value=None)
