# ChatGLM2微调指南 💡
#### ChatGLM3微调[见下方](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/README.md#chatglm3%E5%BE%AE%E8%B0%83%E6%8C%87%E5%8D%97-)
#### AI Agent搭建[见下方](https://github.com/KevinWang676/Bark-Voice-Cloning/tree/main/notebooks#ai-agent-%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97-)
## 1. 环境搭建
### 安装依赖
```
git clone https://github.com/THUDM/ChatGLM2-6B
cd ChatGLM2-6B
pip install -r requirements.txt
pip install rouge_chinese nltk jieba datasets
```
### 下载模型
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
sudo apt install build-essential
git clone https://huggingface.co/THUDM/chatglm2-6b
```

## 2. 准备数据集

自建数据集请参考[train.json](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/train.json)及以下格式：
```json lines
{"prompt": "长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "response": "用电脑能读数据流吗？水温多少", "history": []}
{"prompt": "95", "response": "上下水管温差怎么样啊？空气是不是都排干净了呢？", "history": [["长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "用电脑能读数据流吗？水温多少"]]}
{"prompt": "是的。上下水管都好的", "response": "那就要检查线路了，一般风扇继电器是由电脑控制吸合的，如果电路存在断路，或者电脑坏了的话会出现继电器不吸合的情况！", "history": [["长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "用电脑能读数据流吗？水温多少"], ["95", "上下水管温差怎么样啊？空气是不是都排干净了呢？"]]}
```
分别准备训练数据集 `train.json` 和验证数据集 `dev.json` 并将其上传至 `ChatGLM2-6B` 文件夹下

## 3. 开始训练

在终端运行以下指令，即可开始训练
```shell
bash train_chat.sh
```

**注意**：原 `train_chat.sh` 文件中包含以下代码：
```
PRE_SEQ_LEN=128
LR=1e-2
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file $CHAT_TRAIN_DATA \
    --validation_file $CHAT_VAL_DATA \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir $CHECKPOINT_NAME \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
```
**在开始训练前，需要将其编辑为以下示例代码**：
```
PRE_SEQ_LEN=128
LR=1e-2
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS ptuning/main.py \
    --do_train \
    --train_file train.json \
    --validation_file dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path chatglm2-6b \
    --output_dir output_model \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 600 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN
```

P.S. 以上的 `train_chat.sh` 文件只是一个示例，具体参数设置请根据不同GPU的性能进行调节；ChatGLM2微调[官方教程](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning)

# ChatGLM3微调指南 📒

## 1. 环境搭建
### 安装依赖
```
git clone https://github.com/THUDM/ChatGLM3
cd ChatGLM3
pip install -r requirements.txt
pip install transformers==4.34.0
apt install nvidia-cuda-toolkit
cd finetune_chatmodel_demo
pip install -r requirements.txt
cd ..
```
### 下载模型
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
sudo apt install build-essential
git clone https://huggingface.co/THUDM/chatglm3-6b
```
## 2. 准备数据集

自建数据集请参考[train_linghua_new_v3.json](https://github.com/KevinWang676/Bark-Voice-Cloning/blob/main/notebooks/train_linghua_new_v3.json)及以下格式：
```json
[
  {
    "conversations": [
      {
        "role": "system",
        "content": "<system prompt text>"
      },
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      }, 
       // ... Muti Turn
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      }
    ]
  }
  // ...
]
```


准备训练数据集 `train.json` 并将其上传至 `ChatGLM3` 文件夹下

## 3. 开始训练

在终端运行以下指令，即可开始训练
```shell
bash finetune_chatmodel_demo/scripts/finetune_pt_multiturn.sh
```

**注意**：原 `finetune_pt_multiturn.sh` 文件中包含以下代码：
```
#! /usr/bin/env bash

set -ex

PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
MAX_SEQ_LEN=2048
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=16
MAX_STEP=1000
SAVE_INTERVAL=500

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=tool_alpaca_pt

BASE_MODEL_PATH=THUDM/chatglm3-6b
DATASET_PATH=formatted_data/tool_alpaca.jsonl
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${PRE_SEQ_LEN}-${LR}

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format multi-turn \
    --train_file $DATASET_PATH \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
```
**在开始训练前，需要将其编辑为以下示例代码**：
```
#! /usr/bin/env bash

set -ex

PRE_SEQ_LEN=128
LR=1e-2
NUM_GPUS=1
MAX_SEQ_LEN=2048
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=16
MAX_STEP=700
SAVE_INTERVAL=100

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=linghua_pt

BASE_MODEL_PATH=chatglm3-6b
DATASET_PATH=train.json
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${PRE_SEQ_LEN}-${LR}

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune_chatmodel_demo/finetune.py \
    --train_format multi-turn \
    --train_file $DATASET_PATH \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
```
P.S. 以上的 `finetune_pt_multiturn.sh` 文件只是一个示例，具体参数设置请根据不同GPU的性能进行调节；ChatGLM3微调[官方教程](https://github.com/THUDM/ChatGLM3/tree/main/finetune_chatmodel_demo)


# AI Agent 使用指南 🌟

## 1. 环境搭建
### 安装依赖
```
git clone https://github.com/KevinWang676/modelscope-agent.git
cd modelscope-agent
pip install -r requirements.txt
mv modelscope_agent apps/agentfabric
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
cd apps/agentfabric
```
### API Key设置
```
import os
os.environ["DASHSCOPE_API_KEY"] = "您的DASHSCOPE_API_KEY"
```
或
`export DASHSCOPE_API_KEY=your_api_key`

## 2. 开始使用
```
python app.py
```
