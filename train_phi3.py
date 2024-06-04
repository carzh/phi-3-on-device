from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import onnxruntime.training.api as ort_api
import torch
from datasets import load_dataset
from functools import partial
import time
import matplotlib.pyplot as plt
import numpy as np

artifacts_dir = "artifacts_torch_export_last_two_layers"

state = ort_api.CheckpointState.load_checkpoint(artifacts_dir + '/checkpoint')
# state = ort_api.CheckpointState.load_checkpoint('after_5_epochs_no_bs.ckpt')
training_model = ort_api.Module(artifacts_dir + '/training_model.onnx', state, artifacts_dir + '/eval_model.onnx', device='cuda')
optimizer = ort_api.Optimizer(artifacts_dir + '/optimizer_model.onnx', training_model)

print("=" * 10)
print("Successfully loaded the training session and optimizer")
print("=" * 10)

modelpath="microsoft/Phi-3-mini-4k-instruct"
dataset_name="g-ronimo/oasst2_top1_en"
lr=0.00008      # learning rate
bs=1            # batch size
bs_eval=16      # batch size for evals
ga_steps=16     # gradient acc. steps
epochs=3
max_length=1048      # samples max. length
output_dir="out"
num_layers = 0

optimizer.set_learning_rate(lr)

tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)    # fast tokenizer sometimes ignores added tokens

# tokenizer.add_tokens(["<|im_start|>", "<PAD>"])
# tokenizer.pad_token = "<PAD>"
# tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))

# Load Dataset
dataset = load_dataset(dataset_name)
dataset = dataset["train"].train_test_split(test_size=0.1)

# chatML Template and tokenize dataset
templates=[
    "<|assistant|>\n{msg}<|endoftext|>\n",
    "<|user|>\n{msg}<|endoftext|>\n"
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
        "position_ids": get_position_ids(torch.tensor(attention_mask[:max_length])),
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
        position_ids=sample["position_ids"]
        attention_mask=sample["attention_mask"]

        pad_len=tokens_maxlen-len(input_ids)

        input_ids.extend( pad_len * [tokenizer.pad_token_id] )   
        labels.extend( pad_len * [IGNORE_INDEX] )    
        position_ids.extend( pad_len * [1] )
        attention_mask.extend( pad_len * [0] ) 

    batch={
        "input_ids": torch.tensor( [e["input_ids"] for e in elements] ).numpy(),
        "labels": torch.tensor( [e["labels"] for e in elements] ).numpy(),
        "position_ids": torch.tensor( [e["position_ids"] for e in elements] ).numpy(),
        "attention_mask": torch.tensor( [e["attention_mask"] for e in elements] ).numpy(),
    }

    return batch

dataloader = torch.utils.data.DataLoader(dataset_tokenized["train"], batch_size=bs, shuffle=True, collate_fn = collate)

losses = []

def make_empty_kv(num_layers):
    kvs = []
    for i in range(num_layers):
        # empty array of size batch_size, num_kv_heads, past_seq_length, dimension of the attn heads (kv_head_dim)
        kvs.append(np.empty(shape=(bs, 32, 0, 96), dtype=np.float32))
        # do it twice -- once for key, once for value
        kvs.append(np.empty(shape=(bs, 32, 0, 96), dtype=np.float32))

    return kvs

def trainEpoch(losses):
    training_model.train()
    i = 0
    kvs_input = make_empty_kv(num_layers)
    for batch in dataloader:
        # forward_inputs = [batch["input_ids"], batch["attention_mask"], batch["position_ids"], batch["labels"]] + kvs_input
        forward_inputs = [batch["input_ids"], batch["attention_mask"], batch["position_ids"], batch["labels"]]
        # forward_inputs = [batch["input_ids"], batch["attention_mask"], batch["labels"]] + kvs_input

        t0 = time.monotonic()
        outputs = training_model(*forward_inputs)
        optimizer.step()
        training_model.lazy_reset_grad()
        t1 = time.monotonic()
        loss = float(outputs[0])
        losses.append(loss)
        if i % 20 == 0:
            # print("decoded input ids", tokenizer.batch_decode(forward_inputs[0], skip_special_tokens = True))
            print(i, 'out of', len(dataloader))
            print('time taken for batch ', i, ' out of ', len(dataloader), ': ', f'{t1-t0:.5f}')
            print('loss: ', loss)
            # print('remaining inputs: ', etc)
            # print()
            # print('decoded input ids:', tokenizer.batch_decode(forward_inputs[0], skip_special_tokens=True))
            # print('decoded labels:', tokenizer.batch_decode(forward_inputs[3], skip_special_tokens=True))
        i += 1

for epoch in range(epochs):
    trainEpoch(losses)

x = np.array(range(len(losses)))
y = np.array(losses)
plt.plot(x, y)
plt.show()
plt.savefig('lossgraph.png')

output_names = ["logits"]

def append_kv_output_list(output_names_list, num_layers):
    key_template = "present.{num_layer}.key"
    value_template = "present.{num_layer}.value"
    for i in range(num_layers):
        output_names_list.append(key_template.format(num_layer=i))
        output_names_list.append(value_template.format(num_layer=i))

append_kv_output_list(output_names, num_layers)

print('exporting with this output list:')
print(output_names)
print()
# ort_api.CheckpointState.save_checkpoint(state, f"after_4_epochs_diff_template.ckpt", include_optimizer_state = True)

# training_model.export_model_for_inferencing("exported_model_4_epochs_kv.onnx", output_names)
training_model.export_model_for_inferencing("exported_model_ft_2_layers.onnx", ["logits"])
