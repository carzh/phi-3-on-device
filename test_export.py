from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import onnxruntime.training.api as ort_api
import torch
from datasets import load_dataset
from functools import partial
import time

artifacts_dir = "artifacts"

state = ort_api.CheckpointState.load_checkpoint('after_3_epochs.ckpt')
training_model = ort_api.Module(artifacts_dir + '/training_model.onnx', state, artifacts_dir + '/eval_model.onnx', device='cuda')
optimizer = ort_api.Optimizer(artifacts_dir + '/optimizer_model.onnx', training_model)

print("=" * 10)
print("Successfully loaded the training session and optimizer")
print("=" * 10)

training_model.export_model_for_inferencing("exported_model_3_epochs.onnx", ["logits"])
