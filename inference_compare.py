from transformers import AutoModelForCausalLM, AutoTokenizer
import onnxruntime_genai as og

# prompt = "Why is the sky blue?"
prompt = "<|user|>\nTell me a NASA joke!<|end|>\n<|assistant|>\n"
prompt = "<|user|>\nI am making mayonnaise, it was starting to thicken but now it has become runny and liquid again, is there any way to salvage it?<|end|>\n<|assistant|>\n"

modelpath = "microsoft/Phi-3-mini-4k-instruct"

#############################################################################################################################
# inferencing with the transformers model (before finetuning)
#############################################################################################################################
def inference(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # outputs = model.generate(input_ids, max_new_tokens=2048, do_sample=True, top_k=50, top_p=0.95, eos_token_id=model.config.eos_token_id)
    outputs = model.generate(input_ids, max_new_tokens=400, do_sample=True, top_k=50, top_p=0.95, eos_token_id=model.config.eos_token_id)
    outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs_decoded

tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

model_before_finetune = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    device_map="auto",
    trust_remote_code=True
)

model_before_finetune.resize_token_embeddings(len(tokenizer))
     # pad_to_multiple_of=64)   # phi2 default is 64, see configuration_phi.py
model_before_finetune.config.eos_token_id = tokenizer.eos_token_id

outputs_before_finetuning = inference(model_before_finetune, tokenizer, prompt)

for output in outputs_before_finetuning:
    print(output)

#############################################################################################################################
# inferencing with the onnx model (after finetuning)
#############################################################################################################################

hf_tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

input_ids_np = hf_tokenizer(prompt, return_tensors="np").input_ids
input_ids_list = input_ids_np.tolist()


path_to_model_folder = 'finetune_results'
model = og.Model(path_to_model_folder)
tokenizer = og.Tokenizer(model)

tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
# params.set_search_options({"max_length": 20})
# params.set_search_options(max_length=400)
params.set_search_options(do_sample=True, max_length=400, top_k = 50, top_p = 0.95)
params.input_ids = input_ids_list

generated_output = model.generate(params)
print(generated_output)

output_tokens = generated_output[0]

text = hf_tokenizer.decode(output_tokens, skip_special_tokens=True)
print(text)
