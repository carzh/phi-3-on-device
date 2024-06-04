import onnxruntime.training.api as ort_api

artifacts_dir = "artifacts_torch_export_last_two_layers"

state = ort_api.CheckpointState.load_checkpoint(artifacts_dir + '/checkpoint')
# state = ort_api.CheckpointState.load_checkpoint('after_5_epochs_no_bs.ckpt')
training_model = ort_api.Module(artifacts_dir + '/training_model.onnx', state, artifacts_dir + '/eval_model.onnx', device='cuda')
optimizer = ort_api.Optimizer(artifacts_dir + '/optimizer_model.onnx', training_model)

training_model.export_model_for_inferencing("exported_torch_model_no_ft.onnx", ["logits"])