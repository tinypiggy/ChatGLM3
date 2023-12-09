from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
import time
import os

MODEL_PATH = '/mnt/e/aiModel/chatglm3-6b-32k'
# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
with init_empty_weights():
  model = AutoModel.from_config(config, trust_remote_code=True)

for name, _ in model.named_parameters():
  print(name)
device_map = infer_auto_device_map(model, no_split_module_classes=["GLMBlock"])
print(device_map)
# device_map = {'transformer.word_embeddings': 0, 'transformer.layers.0': 0, 'transformer.layers.1': 0, 'transformer.layers.2': 0, 'transformer.layers.3': 0, 'transformer.layers.4': 0, 'transformer.layers.5': 0, 'transformer.layers.6': 0, 'transformer.layers.7': 0, 'transformer.layers.8': 0, 'transformer.layers.9': 0, 'transformer.layers.10': 0, 'transformer.layers.11': 0, 'transformer.layers.12': 0, 'transformer.layers.13': 0, 'transformer.layers.14': 0, 'transformer.layers.15': 0, 'transformer.layers.16': 0, 'transformer.layers.17': 0, 'transformer.layers.18': 0, 'transformer.layers.19': 0, 'transformer.layers.20': 0, 'transformer.layers.21': 'cpu', 'transformer.layers.22': 'cpu', 'transformer.layers.23': 'cpu', 'transformer.layers.24': 'cpu', 'transformer.layers.25': 'cpu', 'transformer.layers.26': 'cpu', 'transformer.layers.27': 'cpu', 'transformer.final_layernorm': 'cpu', 'lm_head': 'cpu'}
model = load_checkpoint_and_dispatch(model, MODEL_PATH, device_map=device_map, offload_folder="offload", offload_state_dict=True, no_split_module_classes=["GLMBlock"]).eval()

tools = [{'name': 'track', 'description': '追踪指定股票的实时价格', 'parameters': {'type': 'object', 'properties': {'symbol': {'description': '需要追踪的股票代码'}}, 'required': ['symbol']}}, 
         {'name': '/text-to-speech', 'description': '将文本转换为语音', 'parameters':
           {'type': 'object', 'properties': {'text': {'description': '需要转换成语音的文本'}, 'voice': {'description': '要使用的语音类型（男声、女声等）'}, 'speed': {'description': '语音的速度（快、中等、慢等）'}}, 'required': []}}, 
         {'name': '/image_resizer', 'description': '调整图片的大小和尺寸', 
          'parameters': {'type': 'object', 'properties': {'image_file': {'description': '需要调整大小的图片文件'}, 'width': {'description': '需要调整的宽度值'}, 'height': {'description': '需要调整的高度值'}}, 'required': []}},
         {'name': '/foodimg', 'description': '通过给定的食品名称生成该食品的图片', 'parameters': {'type': 'object', 'properties': {'food_name': {'description': '需要生成图片的食品名称'}}, 'required': []}}]
system_item = {"role": "system",
               "content": "Answer the following questions as best as you can. You have access to the following tools:",
               "tools": tools}

def predict(input, history=None):
    print(f'predict started: {time.time()}');
    if history is None:
        history = []
    response, history = model.chat(tokenizer, input, history)
    return response, history

while True:
  history = [system_item]
  text = input(">>用户：")
  response, history = model.chat(tokenizer, text, history)
  print(">>CHatGLM：", response)