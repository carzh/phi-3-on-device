from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import onnxruntime.training.api as ort_api
import torch
from datasets import load_dataset
from functools import partial
import time
import matplotlib.pyplot as plt
import numpy as np

artifacts_dir = "artifacts_kv"

modelpath="microsoft/Phi-3-mini-4k-instruct"
dataset_name="g-ronimo/oasst2_top1_en"
lr=0.00008      # learning rate
bs=1            # batch size
bs_eval=16      # batch size for evals
ga_steps=16     # gradient acc. steps
epochs=4
max_length=1048      # samples max. length
output_dir="out"
num_layers = 32

model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    device_map = "auto",
    trust_remote_code = True
)
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)    # fast tokenizer sometimes ignores added tokens

# Load Dataset
dataset = load_dataset(dataset_name)
dataset = dataset["train"].train_test_split(test_size=0.1)

# chatML Template and tokenize dataset
templates=[
    "<|assistant|>\n{msg}<|end|>\n",
    "<|user|>\n{msg}<|end|>\n"
]
IGNORE_INDEX=-100

def get_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # Shape: (batch_size, sequence_length)
    return position_ids

# tokenize dataset, set input_ids and attention_mask to train on assistant outputs only
def tokenize(input, max_length):
    input_ids, attention_mask, position_ids, labels = [], [], [], []

    for i,msg in enumerate(input["conversation"]):
        isHuman = msg["role"]=="user"
        msg_chatml=templates[isHuman].format(msg=msg["content"])
        msg_tokenized=tokenizer(msg_chatml, truncation=False, add_special_tokens=False)

        input_ids+=msg_tokenized["input_ids"]
        attention_mask+=msg_tokenized["attention_mask"]
        labels+=[IGNORE_INDEX]*len(msg_tokenized["input_ids"]) if isHuman else msg_tokenized["input_ids"]

    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        # "position_ids": get_position_ids(torch.tensor(attention_mask[:max_length])),
        "labels": labels[:max_length],
    }

print('attempting tokenizing dataset')

dataset_tokenized = dataset.map(
    partial(tokenize, max_length=max_length), 
    batched=False, 
    # num_proc=os.cpu_count(),    # multithreaded
    remove_columns=dataset["train"].column_names  # don't need this anymore, we have tokens from here on
)
print('after dataset_tokenized is generated yayy')

# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokens=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokens])

    for i,sample in enumerate(elements):
        input_ids=sample["input_ids"]
        labels=sample["labels"]
        # position_ids=sample["position_ids"]
        attention_mask=sample["attention_mask"]

        pad_len=tokens_maxlen-len(input_ids)

        input_ids.extend( pad_len * [tokenizer.pad_token_id] )   
        labels.extend( pad_len * [IGNORE_INDEX] )    
        # position_ids.extend( pad_len * [1] )
        attention_mask.extend( pad_len * [0] ) 

    batch={
        "input_ids": torch.tensor( [e["input_ids"] for e in elements] ),
        "labels": torch.tensor( [e["labels"] for e in elements] ),
        # "position_ids": torch.tensor( [e["position_ids"] for e in elements] ).numpy(),
        "attention_mask": torch.tensor( [e["attention_mask"] for e in elements] ),
    }

    return batch

def inference(model, tokenizer, prompt):
    formatted_prompt = templates[1].format(msg=prompt)
    formatted_prompt += "\n<|im_start|>assistant"
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=2048, do_sample=True, top_k=50, top_p=0.95, eos_token_id=model.config.eos_token_id)
    outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs_decoded

prompt_for_comparison = "Why is the sky blue?"

outputs_before_finetuning = inference(model, tokenizer, prompt_for_comparison)
print(outputs_before_finetuning[0])


steps_per_epoch=len(dataset_tokenized["train"])//(bs*ga_steps)

print('='*80)

# model_after_finetuning = AutoModelForCausalLM.from_pretrained("/home/carolinezhu/phi-3-on-device/finetuned-phi3")
model_after_finetuning = AutoModelForCausalLM.from_pretrained("finetuned-phi3", trust_remote_code = True)
outputs_after_finetuning = inference(model_after_finetuning, tokenizer, prompt_for_comparison)
print(outputs_after_finetuning[0])