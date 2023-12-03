# 基础方法封装
import os
import shutil
import glob

# 绝对路径获取方法
curPath = os.path.dirname(os.path.abspath(__file__))
def getAbsPath (relativePath):
  joinPath = os.path.join(curPath, relativePath)
  return os.path.normpath(
    os.path.abspath(joinPath)
  )

# 数据集存放路径
datasets_dir = getAbsPath('./datasets')
if not shutil.os.path.exists(datasets_dir):
  shutil.os.makedirs(datasets_dir)

# 获取数据集列表 ----------------------------------------------------
def get_dataset_list():
  contents = os.listdir(datasets_dir)
  sub_dirs = [
    content
    for content in contents
    if os.path.isdir(os.path.join(datasets_dir, content))
  ]
  return sub_dirs

# 小模型存放路径
models_dir = getAbsPath('./models')
if not shutil.os.path.exists(models_dir):
  shutil.os.makedirs(models_dir)

# 获取模型列表 ----------------------------------------------------
def get_model_list():
  contents = os.listdir(models_dir)
  sub_dirs = [
    content
    for content in contents
    if os.path.isdir(os.path.join(models_dir, content))
  ]
  return sub_dirs

# 确保对应的空目录存在
def ensure_empty_dir(dirpath):
  if shutil.os.path.exists(dirpath):
    shutil.rmtree(dirpath)
  shutil.os.makedirs(dirpath)

# 获取目录中的最后一个文件
def get_last_file(dirpath):
  files = glob.glob(os.path.join(dirpath, '*'))
  sorted_files = sorted(files, key=os.path.basename)
  if sorted_files:
    return sorted_files[-1]
  return False
