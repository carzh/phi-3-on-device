# phi-3-on-device
Training Phi-3-Mini-4k with ORT on-device training APIs.

## Instructions
1. Clone [the phi-3-training-export branch of onnxruntime-genai](https://github.com/carzh/onnxruntime-genai/tree/carzh/phi-3-training-export) branch.
2. Create a conda or py environment and pip install the following:
```
onnx==1.15.0
onnxruntime
transformers
numpy==1.24.2
torch
```
3. Run the following commands:
```bash
cd onnnxruntime-genai/src/python/py/models
python builder.py -m microsoft/Phi-3-mini-4k-instruct -o phi-3-test-with-loss-labels -p fp32 -e cpu -c phi-3-test-temp-with-loss-labels
```
4. Clone this repo.
5. Copy the ONNX model file `phi-3-test-with-loss-labels/model.onnx` and the ONNX data file `phi-3-test-with-loss-labels/model.onnx.data` to the folder containing this repo.
6. Create a new conda environment and run the following commands to install latest nightly ORT training package:
```bash
python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0
pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training
```
Also pip install the following:
```
torch
transformers
datasets
matplotlib
ipykernel
```
7. To generate the training artifacts that includes the gradient graph, use the second conda environment. Edit `generate_phi3_artifacts.py` and check that the ONNX model name that is loaded aligns with the name of the ONNX file that you moved to this repo in step 5. Then run `python generate_phi3_artifacts.py`. 
8. Run `python train_phi3.py`. The last step in that script exports the trained model for inferencing.
9. We need to add kv_cache back into the model to make inferencing earlier. Run `python process_exported_inference_model.py [name_of_exported_file] [new_file.onnx]`
10. Move or copy the resulting new file and its corresponding data file (will have the form of `new_file.onnx.data`) to the `finetune_results` folder. The `finetune_results` folder contains ORT GenAI config files for phi-3 model. 
11. Run `inference_compare.py` which will inference with the Phi-3 huggingface model, then, with the same prompt, inference with your finetuned model.