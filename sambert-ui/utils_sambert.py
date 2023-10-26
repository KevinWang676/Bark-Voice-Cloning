# 训练部分实现
import os
import shutil
import uuid
import gradio as gr
from modelscope.models.audio.tts import SambertHifigan
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import TtsTrainType
from modelscope.hub.utils.utils import get_cache_dir

from utils_base import ensure_empty_dir, get_last_file, models_dir, get_model_list
import consts

# 绝对路径获取方法
curPath = os.path.dirname(os.path.abspath(__file__))
def getAbsPath (relativePath):
  joinPath = os.path.join(curPath, relativePath)
  return os.path.normpath(
    os.path.abspath(joinPath)
  )

# 模型训练 ---------------------------------------------------------
# name      - 训练结果(小模型)命名
# steps     - 训练步数
# train_dataset_zip - 数据集zip包路径
def train(name, steps, train_dataset_name):
  # 创建临时目录用于放置 训练结果
  work_dir = getAbsPath(f'./temp/work-{ uuid.uuid4() }')
  ensure_empty_dir(work_dir)

  # 数据集目录
  train_dataset = getAbsPath(f'./datasets/{ train_dataset_name }')

  # 进行训练
  trainer = build_trainer(
    Trainers.speech_kantts_trainer,
    default_args=dict(
      # 指定要finetune的 模型/版本
      model = consts.base_model_id,
      model_revision = consts.base_model_version,

      work_dir = work_dir,            # 指定临时工作目录
      train_dataset = train_dataset,  # 数据集目录

      # 训练参数
      train_type = {
        TtsTrainType.TRAIN_TYPE_SAMBERT: {  # 配置训练AM（sambert）模型
          'train_steps': steps + 1,        # 训练多少个step
          'save_interval_steps': 20,       # 每训练多少个step保存一次checkpoint
          'log_interval': 10               # 每训练多少个step打印一次训练日志
        }
      }
    )
  )
  trainer.train()

  # 挑选需要的文件到结果目录
  target_dir = os.path.join(models_dir, name)
  ensure_empty_dir(target_dir)
  shutil.os.makedirs(os.path.join(target_dir, 'tmp_am', 'ckpt'))
  shutil.os.makedirs(os.path.join(target_dir, 'data', 'se'))

  shutil.copy(
    get_last_file(os.path.join(work_dir, 'tmp_am', 'ckpt')),
    os.path.join(target_dir, 'tmp_am', 'ckpt')
  )
  shutil.copy(
    os.path.join(work_dir, 'tmp_am', 'config.yaml'),
    os.path.join(target_dir, 'tmp_am'),
  )
  shutil.copy(
    os.path.join(work_dir, 'data', 'audio_config.yaml'),
    os.path.join(target_dir, 'data'),
  )
  shutil.copy(
    os.path.join(work_dir, 'data', 'se', 'se.npy'),
    os.path.join(target_dir, 'data', 'se'),
  )

  # 清理文件
  shutil.rmtree(work_dir)
  shutil.rmtree(train_dataset)

  # 返回结果
  return '训练完成', gr.update(choices=get_model_list())

# 模型推理 ---------------------------------------------------------
# name - 使用的小模型名称
# txt - 需要合成音频的文字
def infer(name, txt):
  try:
    base_model_path = os.path.join(get_cache_dir(), consts.base_model_id)
    model_path = os.path.join(models_dir, name)
    custom_infer_abs = {
      'voice_name': 'F7',

      # 小模型部分
      'am_ckpt': os.path.join(model_path, 'tmp_am', 'ckpt'),
      'am_config': os.path.join(model_path, 'tmp_am', 'config.yaml'),
      'audio_config': os.path.join(model_path, 'data', 'audio_config.yaml'),
      'se_file': os.path.join(model_path, 'data', 'se', 'se.npy'),

      # 基础模型部分
      'voc_ckpt': os.path.join(
        base_model_path, 'basemodel_16k', 'hifigan', 'ckpt'
      ),
      'voc_config': os.path.join(
        base_model_path, 'basemodel_16k', 'hifigan', 'config.yaml'
      )
    }

    model = SambertHifigan(
      base_model_path,
      **{ 'custom_ckpt': custom_infer_abs }
    )
    inference = pipeline(task=Tasks.text_to_speech, model=model)
    output = inference(input=txt)

    output_path = f'/tmp/{ uuid.uuid4() }.wav'
    with open(output_path, mode='bx') as f:
      f.write(output['output_wav'])
    return output_path
  except Exception:
    return False

# 删除模型 ---------------------------------------------------------
# name - 删除的小模型名称
def delete_model(name):
  try:
    if not name:
      return gr.update(choices=get_model_list())

    target_dir = os.path.join(models_dir, name)
    shutil.rmtree(target_dir)
    return gr.update(choices=get_model_list(), value=None)
  except Exception:
    return gr.update(choices=get_model_list(), value=None)
