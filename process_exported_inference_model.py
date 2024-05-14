# to get faster results during inferencing & to get the ONNX model to work with genAI api's, we need to add the past key values back into the model.

import onnx


model = onnx.load("exported_model_20_epochs.onnx")

num_layers = 32
num_heads = 32
head_dims = 96

for layer_id in range(num_layers):
    past_key_name = f"past_key_values.{layer_id}.key"
    past_value_name = f"past_key_values.{layer_id}.value"
    present_key_name = f"present.{layer_id}.key"
    present_value_name = f"present.{layer_id}.value"

    past_key_tensor = onnx.helper.make_tensor_value_info(past_key_name, onnx.TensorProto.FLOAT, ["batch_size", num_heads, "seq_len", head_dims])
    past_value_tensor = onnx.helper.make_tensor_value_info(past_value_name, onnx.TensorProto.FLOAT, ["batch_size", num_heads, "seq_len", head_dims])
    present_key_tensor = onnx.helper.make_tensor_value_info(present_key_name, onnx.TensorProto.FLOAT, ["batch_size", num_heads, "seq_len", head_dims])
    present_value_tensor = onnx.helper.make_tensor_value_info(present_value_name, onnx.TensorProto.FLOAT, ["batch_size", num_heads, "seq_len", head_dims])

    model.graph.input.extend([past_key_tensor, past_value_tensor])
    model.graph.output.extend([present_key_tensor, present_value_tensor])

    mha_node_name = f"/model/layers.{layer_id}/attn/MultiHeadAttention"
    mha_node = None
    for node in model.graph.node:
        if node.name == mha_node_name:
            mha_node = node
            break
    mha_node.input.extend([past_key_name, past_value_name])
    mha_node.output.extend([present_key_name, present_value_name])
    print("past key values added to: ", mha_node.name)

onnx.save_model(model, "exported_model_20_epochs_with_past_key_values.onnx")