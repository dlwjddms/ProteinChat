import logging
import esm
from functools import partial
from . import dnabert

import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from argparse import ArgumentParser
import json

from proteinchat.common.registry import registry
from proteinchat.models.blip2 import Blip2Base, disabled_train
from proteinchat.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

from transformers import AutoTokenizer, EsmModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
# from bitsandbytes import LoraConfig

import time

# In a distributed setup, you typically set devices via environment variables.
# For example, use torchrun to assign ranks, and then:
local_rank = int(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
device = torch.device(f"cuda:{local_rank}")

# We'll use FSDP to shard the entire model, so we avoid manual device assignments as much as possible.
# However, if you still want to designate modules to specific GPUs in a two-GPU setup,
# consider moving parts *before* wrapping with FSDP.
cuda_pt = "cuda:1"
cuda_mo = "cuda:1"
cuda_llama = "cuda:1"

# Import FSDP utilities
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Define transformer layer classes for auto-wrap (adjust based on your models)
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


# Create an auto_wrap_policy callable that later receives (module, recurse, nonwrapped_numel)
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={BertLayer, LlamaDecoderLayer}
)

# Define a mixed precision policy (using FP16)
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

@registry.register_model("proteinchat")
class ProteinChat(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "",
    }

    def __init__(
        self,
        freeze_protein_encoder=True,
        freeze_lp=False,
        freeze_llama=True,
        llama_model="",
        embedding_agg=1, 
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.embedding_agg = embedding_agg
        self.freeze_llama = freeze_llama
        
        print('Loading protein encoder')
        # Load ESM protein encoder
        self.protein_encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.protein_tokenizer = alphabet.get_batch_converter()

        print('Loading microorganism encoder')
        # Load DNABERT-based microorganism encoder
        self.microorganism_encoder, bpe = dnabert.pretrained.load_model_and_tokenizer(
            num_labels=461, 
            model_name_or_path='zhihan1996/DNABERT-2-117M', 
            model_max_length=32767
        )
        self.microorganism_tokenizer = bpe

        # Freeze encoders if needed
        if freeze_protein_encoder:
            for name, param in self.protein_encoder.named_parameters():
                param.requires_grad = False
            self.protein_encoder = self.protein_encoder.eval()
            self.protein_encoder.train = disabled_train

            for name, param in self.microorganism_encoder.named_parameters():
                param.requires_grad = False
            self.microorganism_encoder = self.microorganism_encoder.eval()
            self.microorganism_encoder.train = disabled_train
        else:
            self.protein_encoder.train()
            self.microorganism_encoder.train()

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        if low_resource:
            print("Start Low Resource Mode")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                #torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map='auto'
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                #torch_dtype=torch.float16,
            )
            
        # Freeze LLAMA if needed or apply LoRA
        if freeze_llama:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            lora_target_modules = ["q_proj", "v_proj"]
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()

        # Projection layers
        self.glm_llama_proj = nn.Linear(1280, self.llama_model.config.hidden_size)
        self.glm_llama_proj_2 = nn.Linear(768, self.llama_model.config.hidden_size)
 
        if freeze_lp:
            for name, param in self.glm_llama_proj.named_parameters():
                param.requires_grad = False
            for name, param in self.glm_llama_proj_2.named_parameters():
                param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        # Enable gradient checkpointing for transformer modules
        self.llama_model.gradient_checkpointing_enable()
        try:
            self.glm_llama_proj.gradient_checkpointing_enable()
            self.glm_llama_proj_2.gradient_checkpointing_enable()
            
        except Exception as e:
            logging.info("Projection layers do not support gradient checkpointing directly.")
        try:
            self.protein_encoder.gradient_checkpointing_enable()
        except Exception as e:
            logging.info("Protein encoder does not support gradient checkpointing directly.")
        try:
            self.microorganism_encoder.gradient_checkpointing_enable()
        except Exception as e:
            logging.info("Microorganism encoder does not support gradient checkpointing directly.")

        # Wrap the full model in FSDP for sharding
        # We assume that the entire model (including encoders and LLAMA) can be wrapped.
        self = FSDP(
            self,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            device_id=torch.cuda.current_device(), #device,  # ensure FSDP uses the correct device per rank
            use_orig_params=True,
        )

    def encode_protein(self, seqs):
        batch_seqs = [( 'protein', seq ) for seq in seqs]
        batch_labels, batch_strs, batch_tokens = self.protein_tokenizer(batch_seqs)
        batch_tokens = batch_tokens#.to(device)

        # Forward pass through protein encoder
        protein_out = self.protein_encoder(batch_tokens, repr_layers=[33], return_contacts=True)
        protein_embeds = protein_out["representations"][33]#.to(batch_tokens.device)
        if protein_embeds.dtype != self.glm_llama_proj.weight.dtype:
            protein_embeds = protein_embeds#.to(self.glm_llama_proj.weight.dtype)
        inputs_llama = self.glm_llama_proj(protein_embeds.squeeze(dim=2))
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long)#.to(protein_embeds.device)
        return inputs_llama, atts_llama

    def encode_microorganism(self, dnas, dna_labels, mode='mean'):
        batch_dnas, batch_labels = [], []
        for dna, label in zip(dnas, dna_labels):
            batch_labels.append(int(label))
            batch_dnas.append(dna)
        labels, batch_inputs, attention_masks = self.microorganism_tokenizer(batch_dnas, labels=batch_labels)
        batch_inputs = batch_inputs#.to(device)
        labels = torch.tensor(labels)#.to(device)
        attention_masks = attention_masks#.to(device)

        microorganism_out = self.microorganism_encoder(
            input_ids=batch_inputs, attention_mask=attention_masks, labels=labels
        )
        microorganism_embeds = microorganism_out.hidden_states  # [B, seq_len, 768]
        if microorganism_embeds.dtype != self.glm_llama_proj_2.weight.dtype:
            microorganism_embeds = microorganism_embeds#.to(self.glm_llama_proj_2.weight.dtype)
        inputs_llama = self.glm_llama_proj_2(microorganism_embeds.squeeze(dim=2))
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long)#.to(microorganism_embeds.device)
        return inputs_llama, atts_llama
    
    def llama_embed_tokens(self, *args):
        if self.freeze_llama:
            return self.llama_model.model.embed_tokens(*args)
        return self.llama_model.base_model.model.model.embed_tokens(*args)
        
    def prompt_list_wrap(self, img_embeds, img_embeds_2, atts_1, atts_2, prompt):
        if prompt:
            img_embeds = img_embeds#.to(device)
            img_embeds_2 = img_embeds_2#.to(device)
            atts_1 = atts_1#.to(device)
            atts_2 = atts_2#.to(device)
            p_before_lst, p_middle_lst, p_after_lst = [], [], []
            for p in prompt:
                p_before, rest = p.split('<proteinHere>', 1)
                p_middle, p_after = rest.split('<microorganismHere>', 1)
                p_before_lst.append(p_before)
                p_middle_lst.append(p_middle)
                p_after_lst.append(p_after)
            p_before_tokens = self.llama_tokenizer(p_before_lst, return_tensors="pt", add_special_tokens=False)#.to(device)
            p_middle_tokens = self.llama_tokenizer(p_middle_lst, return_tensors="pt", add_special_tokens=True, padding=True)#.to(device)
            p_after_tokens = self.llama_tokenizer(p_after_lst, return_tensors="pt", add_special_tokens=True, padding=True)#.to(device)
            p_before_embeds = self.llama_embed_tokens(p_before_tokens.input_ids)
            p_middle_embeds = self.llama_embed_tokens(p_middle_tokens.input_ids)
            p_after_embeds = self.llama_embed_tokens(p_after_tokens.input_ids)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_middle_embeds, img_embeds_2, p_after_embeds], dim=1)
            atts_img = torch.cat([atts_1, atts_2], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_1

    def forward(self, samples):
        print('!!!!start')
        seqs = samples["seq"]
        seqs = [s.upper().replace('(', '').replace(')', '') for s in seqs]
        protein_embeds, atts = self.encode_protein(seqs)
        dna_seqs = [s.upper() for s in samples['dna_seq']]
        dna_labels = samples['dna_label']
        microorganism_embeds, mo_atts = self.encode_microorganism(dna_seqs, dna_labels)
        img_embeds, atts_img = self.prompt_list_wrap(protein_embeds, microorganism_embeds, atts, mo_atts, samples["prompt"])
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        )#.to(device)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                                   dtype=torch.long, device=device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=device) * self.llama_tokenizer.bos_token_id
        bos = bos#.to(device)
        to_regress_tokens.input_ids = to_regress_tokens.input_ids#.to(device)
        atts_img = atts_img#.to(device)
        bos_embeds = self.llama_embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        to_regress_embeds = self.llama_embed_tokens(to_regress_tokens.input_ids)
        to_regress_tokens.attention_mask = to_regress_tokens.attention_mask#.to(device)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)
        with autocast(device_type="cuda"):
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        logits = torch.argmax(outputs.logits, dim=2)
        loss = outputs.loss
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        llama_model = cfg.get("llama_model")
        freeze_protein_encoder = cfg.get("freeze_protein_encoder", False)
        freeze_lp = cfg.get("freeze_lp", False)
        freeze_llama = cfg.get("freeze_llama", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        embedding_agg = cfg.get("embedding_agg", 1)
        model = cls(
            freeze_protein_encoder=freeze_protein_encoder,
            freeze_lp=freeze_lp,
            freeze_llama=freeze_llama,
            llama_model=llama_model,
            embedding_agg=embedding_agg, 
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )
        stage1_ckpt = cfg.get("stage1_ckpt", "")
        if stage1_ckpt:
            print("Load GLM and LP Checkpoint: {}".format(stage1_ckpt))
            ckpt = torch.load(stage1_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        peft_ckpt = cfg.get("peft_ckpt", "")
        if peft_ckpt:
            print("Load LoRA Checkpoint: {}".format(peft_ckpt))
            ckpt = torch.load(peft_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model

