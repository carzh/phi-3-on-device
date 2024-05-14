# to get faster results during inferencing & to get the ONNX model to work with genAI api's, we need to add the past key values back into the model.

import onnx
import os

if os.path.isfile("exported_model_20_epochs_with_past_key_values.onnx"):
    os.remove("exported_model_20_epochs_with_past_key_values.onnx")

# remove these files if they already exist because otherwise onnx.save_model will append to the data file
if os.path.isfile("exported_model_20_epochs_with_past_key_values.onnx.data"):
    os.remove("exported_model_20_epochs_with_past_key_values.onnx.data")

model = onnx.load("exported_model_20_epochs.onnx")

num_layers = 32
num_heads = 32
head_dims = 96

for layer_id in range(num_layers):
    past_key_name = f"past_key_values.{layer_id}.key"
    past_value_name = f"past_key_values.{layer_id}.value"
    present_key_name = f"present.{layer_id}.key"
    present_value_name = f"present.{layer_id}.value"

    past_key_tensor = onnx.helper.make_tensor_value_info(past_key_name, onnx.TensorProto.FLOAT, ["batch_size", num_heads, "past_sequence_length", head_dims])
    past_value_tensor = onnx.helper.make_tensor_value_info(past_value_name, onnx.TensorProto.FLOAT, ["batch_size", num_heads, "past_sequence_length", head_dims])
    present_key_tensor = onnx.helper.make_tensor_value_info(present_key_name, onnx.TensorProto.FLOAT, ["batch_size", num_heads, "total_sequence_length", head_dims])
    present_value_tensor = onnx.helper.make_tensor_value_info(present_value_name, onnx.TensorProto.FLOAT, ["batch_size", num_heads, "total_sequence_length", head_dims])

    # add new tensors to graph inputs and outputs
    model.graph.input.extend([past_key_tensor, past_value_tensor])
    model.graph.output.extend([present_key_tensor, present_value_tensor])

    # find layer-specific multihead attention node
    mha_node_name = f"/model/layers.{layer_id}/attn/MultiHeadAttention"
    mha_node = None
    for node in model.graph.node:
        if node.name == mha_node_name:
            mha_node = node
            break

    # change input and output of mha node -- need to delete the old inputs and outputs to get rid of the empty string placeholders; otherwise, new inputs
    # and outputs will be appended to the end of the lists and will run into "too many inputs" errors
    print('mha input nodes before modification:', mha_node.input)
    new_inputs = []
    index = 0
    for i in range(6):
        new_inputs.append(mha_node.input[i])

    if len(new_inputs) == 5:
        new_inputs.append("")

    new_inputs.extend([past_key_name, past_value_name])
    print("new inputs:", new_inputs)
    del mha_node.input[:]
    mha_node.input.extend(new_inputs)

    new_outputs = [mha_node.output[0], present_key_name, present_value_name]
    del mha_node.output[:]
    mha_node.output.extend(new_outputs)
    print("past key values added to: ", mha_node.name)
    print()

onnx.save_model(model, "exported_model_20_epochs_with_past_key_values.onnx", save_as_external_data = True, location="exported_model_20_epochs_with_past_key_values.onnx.data")