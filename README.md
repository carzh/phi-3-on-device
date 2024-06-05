# phi-3-on-device
Training Phi-3-Mini-4k with ORT on-device training APIs.

## Instructions
1. Clone this repo & create a conda or py environment and pip install the following:
```
onnx==1.15.0
transformers
numpy==1.24.2
torch
datasets
matplotlib
```
2. Install nightly version of ONNXRuntime Training, following the instructions from [here](https://onnxruntime.ai/getting-started) (select "On-device training" -> Platform of choice -> "Python" -> Hardware acceleration of choice -> "Nightly". Instructions from the ORT website for installing Nightly version of ONNXRuntime Training should trump the instructions below.)
```
python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0
pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training
```
3. Export the forward graph version of the model using the following script. Currently, this script will export every single model weight as a separate file, so it will generate about 100+ files.
```
python torch_onnx_export.py
```
4. Run the artifact generation script.
```
python generate_phi3_artifacts.py
```
5. Train the model! This script will also generate a .png of the loss graph as well export the model for inferencing. This exported model will consist of two parts: the .ONNX model file, and then a .ONNX.data file that will contain all the associated weights of the model.
```
python train_phi3.py
```
6. Run the inferencing script. Optionally pass in a boolean argument for if you want naive streaming.
```
python inference_compare.py True
```
