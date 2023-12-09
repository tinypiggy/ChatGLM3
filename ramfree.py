from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
import time

MODEL_PATH = '/mnt/e/aiModel/chatglm3-6b-32k'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
with init_empty_weights():
  model = AutoModel.from_config(config, trust_remote_code=True)

for name, _ in model.named_parameters():
  print(name)
device_map = infer_auto_device_map(model, no_split_module_classes=["GLMBlock"])
print(device_map)
# device_map = {'transformer.word_embeddings': 0, 'transformer.layers.0': 0, 'transformer.layers.1': 0, 'transformer.layers.2': 0, 'transformer.layers.3': 0, 'transformer.layers.4': 0, 'transformer.layers.5': 0, 'transformer.layers.6': 0, 'transformer.layers.7': 0, 'transformer.layers.8': 0, 'transformer.layers.9': 0, 'transformer.layers.10': 0, 'transformer.layers.11': 0, 'transformer.layers.12': 0, 'transformer.layers.13': 0, 'transformer.layers.14': 0, 'transformer.layers.15': 0, 'transformer.layers.16': 0, 'transformer.layers.17': 0, 'transformer.layers.18': 0, 'transformer.layers.19': 0, 'transformer.layers.20': 0, 'transformer.layers.21': 'cpu', 'transformer.layers.22': 'cpu', 'transformer.layers.23': 'cpu', 'transformer.layers.24': 'cpu', 'transformer.layers.25': 'cpu', 'transformer.layers.26': 'cpu', 'transformer.layers.27': 'cpu', 'transformer.final_layernorm': 'cpu', 'lm_head': 'cpu'}
model = load_checkpoint_and_dispatch(model, MODEL_PATH, device_map=device_map, offload_folder="offload", offload_state_dict=True, no_split_module_classes=["GLMBlock"]).half()

def predict(input, history=None):
    print(f'predict started: {time.time()}');
    if history is None:
        history = []
    response, history = model.chat(tokenizer, input, history)
    return response, history

while True:
  history = None
  text = input(">>用户：")
  response, history = model.chat(tokenizer, text, history)
  print(">>CHatGLM：", response)