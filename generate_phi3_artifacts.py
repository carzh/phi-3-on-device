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

onnx_model_path = "torch_onnx_export_phi3.onnx"

onnx_model = onnx.load(onnx_model_path, load_external_data=True)

# build the list of params that will be trained (marked as requires_grad)
# and list of params that will be frozen / not trained (marked as frozen_params)
requires_grad = [] 
frozen_params = []

matches = ["layers.32", "lm_head", "layers.31", "layers.30"]

# cycle through onnx model graph initializers and check for parameters that
# match the layers that we want to train
for param in onnx_model.graph.initializer:
    if any(match in param.name for match in matches):
        print(param.name)
        requires_grad.append(param.name)
    else:
        frozen_params.append(param.name)

# call generate_artifacts API. Since the torch_onnx_export_phi3.onnx graph
# already has the loss node attached, no need to add a new one.
artifacts.generate_artifacts(
    onnx_model,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
    artifact_directory="artifacts_torch_export_last_two_layers",
    optimizer=artifacts.OptimType.AdamW,
)
