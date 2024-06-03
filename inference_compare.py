from transformers import AutoModelForCausalLM, AutoTokenizer
import onnxruntime_genai as og
import torch
import numpy as np
import onnxruntime

# prompt = "Why is the sky blue?"
prompt = "<|user|>\nTell me a NASA joke!<|endoftext|>\n<|assistant|>\n"
prompt = "<|user|>\nI am making mayonnaise, it was starting to thicken but now it has become runny and liquid again, is there any way to salvage it?<|endoftext|>\n<|assistant|>\n"
prompt = "<|user|>\nWhy Aristotelian view of physics (impetus and stuff) is wrong?<|endoftext|>\n<|assistant|>\n"

modelpath = "microsoft/Phi-3-mini-4k-instruct"

max_tokens = 800

# #############################################################################################################################
# # inferencing with the transformers model (before finetuning)
# #############################################################################################################################

# def inference(model, tokenizer, prompt):
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     # outputs = model.generate(input_ids, max_new_tokens=2048, do_sample=True, top_k=50, top_p=0.95, eos_token_id=model.config.eos_token_id)
#     outputs = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=True, top_k=1, eos_token_id=model.config.eos_token_id)
#     outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return outputs_decoded

# tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

# model_before_finetune = AutoModelForCausalLM.from_pretrained(
#     modelpath,    
#     device_map="auto",
#     trust_remote_code=True
# )

# model_before_finetune.resize_token_embeddings(len(tokenizer))
#      # pad_to_multiple_of=64)   # phi2 default is 64, see configuration_phi.py
# model_before_finetune.config.eos_token_id = tokenizer.eos_token_id

# outputs_before_finetuning = inference(model_before_finetune, tokenizer, prompt)

# print()
# for output in outputs_before_finetuning:
#     print(output)
#     print()

#############################################################################################################################
# # inferencing with the onnx model with genai apis (after finetuning)
# #############################################################################################################################

# hf_tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

# input_ids_np = hf_tokenizer(prompt, return_tensors="np").input_ids
# input_ids_list = input_ids_np.tolist()


# path_to_model_folder = 'finetune_results'
# model = og.Model(path_to_model_folder)
# tokenizer = og.Tokenizer(model)

# tokens = tokenizer.encode(prompt)

# params = og.GeneratorParams(model)
# # params.set_search_options({"max_length": 20})
# # params.set_search_options(max_length=400)
# params.set_search_options(do_sample=True, max_length=800, top_k = 50, top_p = 0.95)
# params.input_ids = input_ids_list

# generated_output = model.generate(params)
# print("=" * 10)
# print("after finetuning")
# print("=" * 10)

# output_tokens = generated_output[0]

# text = hf_tokenizer.decode(output_tokens, skip_special_tokens=True)
# print(text)

#############################################################################################################################
# inferencing with the onnx model with genai apis (after finetuning)
#############################################################################################################################

hf_tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

inference_onnx_model = onnxruntime.InferenceSession("finetune_results/exported_model_noft.onnx")

next_token = ""
accumulated_promptresponse = prompt
output_names = ["logits"]

def get_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # Shape: (batch_size, sequence_length)
    return position_ids.numpy()

iter = 0
print()
print()
print(accumulated_promptresponse)
accumulated_ids = hf_tokenizer(accumulated_promptresponse, return_tensors = "np")['input_ids'][0].tolist()
while accumulated_promptresponse[-1] != "<|end|>":
    if iter > max_tokens:
        break
    input_dict = hf_tokenizer(accumulated_promptresponse, return_tensors = "np")
    input_dict["position_ids"] = get_position_ids(torch.from_numpy(input_dict["attention_mask"]))

    # convert to OrtValues since normal run with np values isn't working for mysterious reasons
    for key in input_dict.keys():
        input_dict[key] = onnxruntime.OrtValue.ortvalue_from_numpy(input_dict[key])

    # run returns a list of ort values, take the first one and convert to numpy array
    results = inference_onnx_model.run_with_ort_values(output_names, input_dict)[0].numpy()
    last_token_distribution = results[0][-1]

    next_token_id = last_token_distribution.argmax()
    accumulated_ids.append(next_token_id)

    next_token = hf_tokenizer.decode(next_token_id, skip_special_tokens = True)
    print(next_token, " ", sep="", end="", flush = True)
    # let the huggingface tokenizer handle when to add spaces between tokens
    accumulated_promptresponse = hf_tokenizer.batch_decode([accumulated_ids], skip_special_tokens = True)

    iter += 1

print()
print("outputs:", hf_tokenizer.batch_decode([accumulated_ids], skip_special_tokens = True))
