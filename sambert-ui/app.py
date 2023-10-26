import gradio as gr

import consts
from utils_base import get_dataset_list, get_model_list
from utils_label import auto_label, delete_dataset
from utils_sambert import train, infer, delete_model

def refresh():
  return gr.update(choices=get_dataset_list()), gr.update(choices=get_model_list())

# gradio server ---------------------------
with gr.Blocks() as server:
  # 面板说明
  gr.Markdown('# Sambert 语音 训练/推理 服务')

  # 标记
  gr.Markdown('## 数据标注')
  with gr.Row():
    label_audio_input = gr.Audio(type='filepath', label='请上传一段音频')
    label_name_input = gr.Textbox(label='数据集命名')
    label_status_output = gr.Textbox(label='打标状态')
    label_btn = gr.Button('开始打标')

  # 训练
  gr.Markdown('## 训练')
  with gr.Row():
    train_dataset_input = gr.Radio(label='数据集选择', choices=get_dataset_list())
    train_name_input = gr.Textbox(label='模型保存名称')
    train_steps_input = gr.Number(label='训练步数, 需要为20的整数倍')
    train_status_output = gr.Text(label='训练状态')
    train_btn = gr.Button('训练')
    dataset_delete_btn = gr.Button('删除数据集', variant='stop')

  # 推理
  # 参考 https://mdnice.com/writing/a40f4bcd3b3e40d8931512186982b711
  # 使用 gr.update 实现对应的联动效果
  gr.Markdown('## 生成')
  with gr.Row():
    infer_name_input = gr.Radio(label='推理模型选择', choices=get_model_list())
    infer_txt_input = gr.Textbox(label='文本', lines=3)
    infer_audio_output = gr.Audio(type='filepath', label='合成的音频')
    infer_btn = gr.Button('生成语音', variant='primary')
    model_delete_btn = gr.Button('删除模型', variant='stop')

  # 逻辑部分
  label_btn.click(
    auto_label,
    inputs=[label_audio_input, label_name_input],
    outputs=[label_status_output, train_dataset_input]
  )

  dataset_delete_btn.click(
    delete_dataset,
    inputs=train_dataset_input,
    outputs=[train_dataset_input]
  )

  train_btn.click(
    train,
    inputs=[train_name_input, train_steps_input, train_dataset_input],
    outputs=[train_status_output, infer_name_input]
  )

  infer_btn.click(
    infer,
    inputs=[infer_name_input, infer_txt_input],
    outputs=[infer_audio_output]
  )

  model_delete_btn.click(
    delete_model,
    inputs=infer_name_input,
    outputs=[infer_name_input]
  )

  server.load(
    refresh,
    inputs=[],
    outputs=[train_dataset_input, infer_name_input]
  )

server.launch(server_port=consts.port, server_name='0.0.0.0')
