from transformers import AutoModelForCausalLM, AutoTokenizer
import onnxruntime_genai as og
import torch
import numpy as np
import onnxruntime
import sys

streaming_mode = False

if len(sys.argv) > 0:
    streaming_mode = bool(sys.argv[1])
    print("set streaming mode to: ", streaming_mode)

# prompt = "Why is the sky blue?"
# prompt = "<|user|>\nTell me a NASA joke!<|endoftext|>\n<|assistant|>\n"
# prompt = "<|user|>\nI am making mayonnaise, it was starting to thicken but now it has become runny and liquid again, is there any way to salvage it?<|endoftext|>\n<|assistant|>\n"
prompt = "<|user|>\nWhy Aristotelian view of physics (impetus and stuff) is wrong?<|end|>\n<|assistant|>\n"

modelpath = "microsoft/Phi-3-mini-4k-instruct"
max_tokens = 800
hf_tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

def get_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # Shape: (batch_size, sequence_length)
    return position_ids.numpy()

def inference(onnx_model_path, tokenizer, prompt, naive_streaming):
    inference_model = onnxruntime.InferenceSession(onnx_model_path)

    next_token_id = -1
    accumulated_prompt = prompt
    # put the accumulated prompt through the tokenizer, take just the input ids from the resulting batch encoding, flatten, 
    # convert to list
    accumulated_ids = tokenizer(accumulated_prompt, return_tensors = "np")["input_ids"][0].tolist()
    output_names = ["logits"]
    generated_tokens = 0

    if naive_streaming:
        print(accumulated_prompt)
        print()

    while next_token_id != tokenizer.eos_token_id:
        if generated_tokens >= max_tokens:
            break

        input_batch_encoding = hf_tokenizer(accumulated_prompt, return_tensors = "np")

        # adding position ids directly to batch encoding will result in type error
        input_dict = {
            "input_ids": input_batch_encoding["input_ids"],
            "attention_mask": input_batch_encoding["attention_mask"],
            "position_ids": get_position_ids(torch.from_numpy(input_batch_encoding["attention_mask"]))
        }

        # run returns a list of ort values, take the first one and convert to numpy array -> resulting shape = 
        # [batch_size, sequence_length, vocab_size]
        results_np = inference_model.run(output_names, input_dict)[0].numpy()
        # slice into [vocab_size]
        last_token_distribution = results_np[0][-1]

        next_token_id = last_token_distribution.argmax()
        accumulated_ids.append(next_token_id)

        accumulated_prompt = hf_tokenizer.batch_decode([accumulated_ids], skip_special_tokens = True)

        i += 1

        if naive_streaming:
            print(hf_tokenizer.decode(next_token_id), sep = " ", end = "", flush = True)
        
    return accumulated_ids, accumulated_prompt

pre_ft_acc_ids, pre_ft_acc_text = inference("exported_torch_model_no_ft.onnx", hf_tokenizer, prompt, streaming_mode)
post_ft_acc_ids, post_ft_acc_text = inference("exported_model_ft_2_layers.onnx", hf_tokenizer, prompt, streaming_mode)
print()
print()
print("before finetune: ", pre_ft_acc_text)
print("=" * 50)
print("after finetune: ", post_ft_acc_text)
