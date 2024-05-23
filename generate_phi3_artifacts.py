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
# onnx_model_path = "phi3-3.8b-4k-fp32-cpu.onnx"
# onnx_model_path = "phi3-mini-4k-instruct-cpu-int4-rtn-block-32.onnx"

# torch_onnx_model = onnx.load(onnx_model_path)

# inferenced_model = onnx.shape_inference.infer_shapes(torch_onnx_model)

# print("after shape inference")
# trilu_input_name = None
# for node in inferenced_model.graph.node:
#     if "Trilu" in node.name:
#         print(node)
#         print(node.input[0])
#         trilu_input_name = node.input[0]
#         break

# for value_info in inferenced_model.graph.value_info:
#     if value_info.name == trilu_input_name:
#         print(value_info)
    

if True:
    # if valid onnx model, will load into an inference session without any error. 
    # TODO: move this check to within the generate_artifacts call
    # validation_check = InferenceSession(inferenced_model.SerializeToString())
    validation_check = InferenceSession(onnx_model_path)
    print("Model is valid yippeee")
onnx_model = onnx.load(onnx_model_path, load_external_data=True)
requires_grad = [] 
frozen_params = []

matches = ["layers.32", "lm_head", "model.layers.31.mlp"]
# matches = ["layers.0", "layers.32", "lm_head", "model.layers.31.mlp"]

for param in onnx_model.graph.initializer:
    if any(match in param.name for match in matches):
        print(param.name)
        requires_grad.append(param.name)
    else:
        frozen_params.append(param.name)
        # print(param.name)
        # requires_grad.append(param.name)
artifacts.generate_artifacts(
    onnx_model,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
    artifact_directory="artifacts_torch_export",
    optimizer=artifacts.OptimType.AdamW,
    ort_format=False,
)
