from transformers import AutoModelForCausalLM, AutoTokenizer
import onnxruntime_genai as og

# prompt = "Why is the sky blue?"
prompt = "<|user|>Tell me a NASA joke!<|end|><|assistant|>"

modelpath = "microsoft/Phi-3-mini-4k-instruct"

templates = [
    "<|im_start|>assistant\n{msg}<|im_end|>",
    "<|im_start|>user\n{msg}<|im_end|>"
]

#############################################################################################################################
# inferencing with the transformers model (before finetuning)
#############################################################################################################################
def inference(model, tokenizer, prompt):
    formatted_prompt = templates[1].format(msg=prompt)
    # formatted_prompt += "\n<|im_start|>assistant"
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids
    # outputs = model.generate(input_ids, max_new_tokens=2048, do_sample=True, top_k=50, top_p=0.95, eos_token_id=model.config.eos_token_id)
    outputs = model.generate(input_ids, max_new_tokens=5, do_sample=True, top_k=50, top_p=0.95, eos_token_id=model.config.eos_token_id)
    outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs_decoded

# tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

# tokenizer.add_tokens(["<|im_start|>", "<PAD>"])
# tokenizer.pad_token = "<PAD>"
# tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))

# model_before_finetune = AutoModelForCausalLM.from_pretrained(
#     modelpath,    
#     device_map="auto",
#     trust_remote_code=True
# )

# model_before_finetune.resize_token_embeddings(len(tokenizer))
#     # pad_to_multiple_of=64)   # phi2 default is 64, see configuration_phi.py
# model_before_finetune.config.eos_token_id = tokenizer.eos_token_id

# outputs_before_finetuning = inference(model_before_finetune, tokenizer, prompt)

# for output in outputs_before_finetuning:
#     print(output)

#############################################################################################################################
# inferencing with the onnx model (after finetuning)
#############################################################################################################################

path_to_model_folder = 'finetune_results'
model = og.Model(path_to_model_folder)
tokenizer = og.Tokenizer(model)

tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
# params.set_search_options({"max_length": 20})
params.set_search_options(max_length=400)
# params.set_search_options(max_length=400, top_k = 50, top_p = 0.95)
params.input_ids = tokens

generated_output = model.generate(params)
print(generated_output)

output_tokens = generated_output[0]

text = tokenizer.decode(output_tokens)
print(text)