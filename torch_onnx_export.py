from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from functools import partial
import torch

modelpath = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

pt_model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    device_map="auto",
    trust_remote_code=True
)

# =============================================================================
# make dummy data
# =============================================================================
dataset_name="g-ronimo/oasst2_top1_en"
IGNORE_INDEX = -100
bs=3            # batch size
max_length = 1048

templates=[
    "<|assistant|>\n{msg}<|end|>\n",
    "<|user|>\n{msg}<|end|>\n"
]

dataset = load_dataset(dataset_name)
dataset = dataset["train"].train_test_split(test_size=0.1)

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
        "input_ids": torch.tensor( [e["input_ids"] for e in elements] ),
        "labels": torch.tensor( [e["labels"] for e in elements] ),
        "position_ids": torch.tensor( [e["position_ids"] for e in elements] ),
        "attention_mask": torch.tensor( [e["attention_mask"] for e in elements] ),
    }

    return batch

dataloader = torch.utils.data.DataLoader(dataset_tokenized["train"], batch_size=bs, shuffle=True, collate_fn = collate)

one_batch = next(iter(dataloader))

print("shape of input ids", one_batch["input_ids"].shape)
print("shape of attention mask", one_batch["attention_mask"].shape)
print("shape of position ids", one_batch["position_ids"].shape)
print("shape of labels", one_batch["labels"].shape)

# =============================================================================
# export model
# =============================================================================

# class FlatModel(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, *local_inputs):
#         return self.model(inputs.input_ids, inputs.attention_mask, inputs.token_type_ids, labels=labels)

# flat_pt_model = FlatModel(pt_model)

torch.onnx.export(
    pt_model,
    # pass in None for the past key values and for the input embeds
    (one_batch["input_ids"], one_batch["attention_mask"], one_batch["position_ids"], None, None, one_batch["labels"]),
    "torch_onnx_export_phi3.onnx",
    input_names = ["input_ids", "attention_mask", "position_ids", "labels"],
    output_names = ["loss", "logits"],
    dynamic_axes= {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
        "labels": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"}
    },
    export_params = True,
    opset_version = 17,
    do_constant_folding = False,
    training = torch.onnx.TrainingMode.TRAINING
)