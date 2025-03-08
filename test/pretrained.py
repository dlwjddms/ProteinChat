'''
        python train.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/EMP/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_EMP_${data}_seed${seed} \
            --model_max_length 128 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert2 \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
'''
import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 #data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()
        self.kmer = kmer
        self.tokenizer = tokenizer
        # load data from the disk

    
    def __call__(self, seqs, labels):
        '''
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        '''
        if self.kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {self.kmer}-mer as input...")
            seqs = load_or_generate_kmer('_', seqs, self.kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))
        return self.labels, self.input_ids, self.attention_mask 

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])



def load_model_and_tokenizer(num_labels, model_name_or_path='zhihan1996/DNABERT-2-117M', use_lora=False, lora_r=8, lora_alpha=21, lora_dropout=0.05, lora_target_modules="query,value", cache_dir=None, model_max_length=128, kmer=-1):

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token
        # load model

    _tokenizer = SupervisedDataset(tokenizer=tokenizer, kmer=kmer)
    
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        num_labels=num_labels,
        trust_remote_code=True,
    )

    # configure LoRA
    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(lora_target_modules.split(",")),
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
    
    return model, _tokenizer


import json

# Read data from JSON file
with open('/home/je/amp-foundation-model/data/train_sequence.json', 'r') as file:
    data = json.load(file)

# Specify the column (key) to pick
column_name = 'micoorganism_sequence'

# Extract the specified column
column_data = [item[column_name] for item in data]
labels = list(set(column_data))
column_label = [ labels.index(x) for x in column_data] 
length = [len(x) for x in column_data]

# Print the extracted column
print(len(labels), max(length))

torch.cuda.set_device(1) 
print(torch.cuda.current_device())
print(os.environ['CUDA_VISIBLE_DEVICES'])
microorganism_encoder, microorganism_tokenizer = load_model_and_tokenizer(num_labels=len(labels) ,model_name_or_path='zhihan1996/DNABERT-2-117M', model_max_length=max(length))
microorganism_encoder = microorganism_encoder.to(torch.cuda.current_device())
batch_dnas, batch_labels = column_data[:2], column_label[:2]
labels , batch_inputs, attention_masks= microorganism_tokenizer(batch_dnas, labels=batch_labels)#["input_ids"]
batch_inputs = batch_inputs.to(torch.cuda.current_device())
labels = torch.tensor(labels).to(torch.cuda.current_device())
attention_masks = attention_masks.to(torch.cuda.current_device())

print(f'!!!!! batch_tokens {batch_inputs.shape}')
microorganism_embeds = microorganism_encoder(input_ids=batch_inputs, attention_mask=attention_masks, labels=labels) # [1, sequence_length, 768]

print(f'!!!! microorganism_embeds {microorganism_embeds.hidden_states.shape}')
microorganism_embeds = microorganism_embeds.hidden_states
# embedding with mean pooling
mode = 'mean'
if mode == 'mean': microorganism_embeds = torch.mean(microorganism_embeds, dim=1)
else: microorganism_embeds = torch.max(microorganism_embeds[0], dim=0)[0]
print(f"@@@@@@@@@ hidden_states {microorganism_embeds.shape}")

