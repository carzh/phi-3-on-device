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

prompt = "<|user|>\nWhy is the sky blue?<|endoftext|>\n<|assistant|>\n"
# prompt = "<|user|>\nTell me a NASA joke!<|endoftext|>\n<|assistant|>\n"
# prompt = "<|user|>\nI am making mayonnaise, it was starting to thicken but now it has become runny and liquid again, is there any way to salvage it?<|endoftext|>\n<|assistant|>\n"
# prompt = "<|user|>\nWhy Aristotelian view of physics (impetus and stuff) is wrong?<|end|>\n<|assistant|>\n"

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
    accumulated_ids = tokenizer(prompt, return_tensors = "np")["input_ids"]
    output_names = ["logits"]
    generated_tokens = 0

    if naive_streaming:
        print(prompt)
        print()

    while next_token_id != tokenizer.eos_token_id:
        if generated_tokens >= max_tokens:
            break

        input_dict = {
            "input_ids": accumulated_ids,
            "attention_mask": torch.ones(size=accumulated_ids.shape, dtype=torch.int64).numpy(),
        }
        input_dict["position_ids"] = get_position_ids(torch.from_numpy(input_dict["attention_mask"]))

        # run returns a list of numpy values, take the first one -> resulting shape = 
        # [batch_size, sequence_length, vocab_size]
        results_np = inference_model.run(output_names, input_dict)[0]
        # slice into [vocab_size]
        last_token_distribution = results_np[0][-1]

        next_token_id = last_token_distribution.argmax()
        accumulated_ids = np.append(accumulated_ids, [[next_token_id]], axis=1)

        generated_tokens += 1

        if naive_streaming:
            print(hf_tokenizer.decode(next_token_id), " ", sep = "", end = "", flush = True)
        
    accumulated_prompt = hf_tokenizer.batch_decode([accumulated_ids], skip_special_tokens = True)
    return accumulated_ids, accumulated_prompt

pre_ft_acc_ids, pre_ft_acc_text = inference("exported_torch_model_no_ft.onnx", hf_tokenizer, prompt, streaming_mode)
post_ft_acc_ids, post_ft_acc_text = inference("exported_model_ft_2_layers.onnx", hf_tokenizer, prompt, streaming_mode)
print()
print()
print("before finetune: ", pre_ft_acc_text[0])
print("=" * 50)
print("after finetune: ", post_ft_acc_text[0])
