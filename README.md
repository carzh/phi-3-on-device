# phi-3-on-device
Training Phi-3-Mini-4k with ORT on-device training APIs.

## Instructions
1. Clone [this](https://github.com/carzh/onnxruntime-genai/tree/carzh/phi-3-training-export) branch.
2. pip install the following:
```
onnx==1.15.0
onnxruntime
transformers
numpy==1.24.2
torch
```
3. Run the following commands
```
cd onnnxruntime-genai/src/python/py/models
python builder.py -m microsoft/Phi-3-mini-4k-instruct -o phi-3-test-with-loss-labels -p fp32 -e cpu -c phi-3-test-temp-with-loss-labels
```
4. Clone this repo.
5. Copy the ONNX model file `phi-3-test-with-loss-labels/model.onnx` and the ONNX data file `phi-3-test-with-loss-labels/model.onnx.data` to the folder containing this repo.
