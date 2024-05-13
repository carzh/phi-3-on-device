from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from onnxruntime.training import artifacts
import onnxruntime.training.api as ort_api
from onnxruntime import InferenceSession
import torch
import onnx
import transformers
import numpy as np
from datasets import load_dataset
from functools import partial
import os

onnx_model_path = "model.onnx"
# onnx_model_path = "phi3-3.8b-4k-fp32-cpu.onnx"
# onnx_model_path = "phi3-mini-4k-instruct-cpu-int4-rtn-block-32.onnx"

if True:
    # if valid onnx model, will load into an inference session without any error. 
    # TODO: move this check to within the generate_artifacts call
    validation_check = InferenceSession(onnx_model_path)
    print("Model is valid yippeee")
onnx_model = onnx.load(onnx_model_path, load_external_data=False)
requires_grad = [] 
frozen_params = []

matches = ["layers.32", "lm_head", "model.layers.31.mlp"]

for param in onnx_model.graph.initializer:
    if any(match in param.name for match in matches):
        print(param.name)
        requires_grad.append(param.name)
    else:
        frozen_params.append(param.name)
artifacts.generate_artifacts(
    onnx_model,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
    artifact_directory="artifacts_matmul",
    optimizer=artifacts.OptimType.AdamW,
    ort_format=False,
)
