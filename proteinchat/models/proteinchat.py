import logging
import esm
from . import dnabert

import torch
#torch.distributed.destroy_process_group()

from torch.cuda.amp import autocast as autocast
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

cuda_pt = "cuda:0"
cuda_mo = "cuda:0"
cuda_llama = "cuda:0"

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
        max_txt_len=512,
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

        self.protein_encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.protein_tokenizer = alphabet.get_batch_converter()
        ##########
        # JE
        ##########
        # 461 32767
        self.microorganism_encoder, bpe = dnabert.pretrained.load_model_and_tokenizer(num_labels=533, model_name_or_path='zhihan1996/DNABERT-2-117M', model_max_length=512) #, use_lora=True)
        self.microorganism_tokenizer = bpe

        '''
        #  model = nn.DataParallel(model)
        self.microorganism_encoder = nn.DataParallel(self.microorganism_encoder)
        #model.to(device)
        self.microorganism_encoder.to(cuda_mo)
        self.protein_encoder= nn.DataParallel(self.protein_encoder)
        self.protein_encoder.to(cuda_mo)
        '''

        if freeze_protein_encoder:
            for name, param in self.protein_encoder.named_parameters():
                param.requires_grad = False
            self.protein_encoder = self.protein_encoder.eval()
            self.protein_encoder.train = disabled_train
            logging.info("freeze protein encoder")
            ##########
            # JE
            ##########
            for name, param in self.microorganism_encoder.named_parameters():
                param.requires_grad = False
            self.microorganism_encoder = self.microorganism_encoder.eval()
            self.microorganism_encoder.train = disabled_train
        else:
            self.protein_encoder = self.protein_encoder.train()
            self.protein_encoder=self.protein_encoder.to(cuda_pt)#.half()
            ##########
            # JE
            ##########
            for name, param in self.microorganism_encoder.named_parameters():
                param.requires_grad = False
            self.microorganism_encoder = self.microorganism_encoder.eval()
            self.microorganism_encoder.train = disabled_train

            #self.microorganism_encoder = self.microorganism_encoder.train()
            #self.microorganism_encoder=self.microorganism_encoder.to(cuda_mo)#.half()
        
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)

        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        if self.low_resource:
            print("Start Low Resource Mode")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map='auto'
                # device_map={'': device_8bit}
            ).to(cuda_llama)#.half()
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                ).to(cuda_llama)#.half()

        if freeze_llama:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            lora_target_modules: List[str] = ["q_proj", "v_proj"]
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

        self.glm_llama_proj = nn.Linear(
            1280, self.llama_model.config.hidden_size
        ).to(cuda_pt)#.half()
        self.glm_llama_proj_2 = nn.Linear(
            768, self.llama_model.config.hidden_size
        ).to(cuda_mo)#.half()
 
        if freeze_lp:
            for name, param in self.glm_llama_proj.named_parameters():
                param.requires_grad = False
        if True:
            for name, param in self.glm_llama_proj_2.named_parameters():
                param.requires_grad = False
 
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        ####
        # JE
        ####
        '''
        Gradient checkpointing trades extra computation for lower memory usage by not storing all intermediate activations. Instead, it will recompute them during the backward pass as needed
        self.llama_model.gradient_checkpointing_enable()
        self.glm_llama_proj.gradient_checkpointing_enable()
        self.glm_llama_proj_2.gradient_checkpointing_enable()
        self.protein_encoder.gradient_checkpointing_enable()
        self.microorganism_encoder.gradient_checkpointing_enable()
        '''
    def sanitize_protein_sequence(self, seq):
        # Define the valid amino acid characters.
        # Including "X" so that any preexisting X remains unchanged.
        valid_chars = set("ACDEFGHIKLMNPQRSTVWYX")
        # Convert sequence to uppercase for consistency
        seq = seq.upper()
        # Replace any character not in the valid set with "X"
        sanitized = "".join([ch if ch in valid_chars else "X" for ch in seq])
        return sanitized

    def encode_protein(self, seqs):
        batch_seqs = []
        for seq in seqs:
            seq = self.sanitize_protein_sequence(seq)
            #seq = seq.replace('J', 'X')
            batch_seqs.append(('protein', seq)) # QQQ Why do we need 'protein'
        batch_labels, batch_strs, batch_tokens = self.protein_tokenizer(batch_seqs)
        batch_tokens = batch_tokens.to(cuda_pt) #.to(torch.cuda.current_device())

        self.protein_encoder=self.protein_encoder.to(cuda_pt)
        self.glm_llama_proj = self.glm_llama_proj.to(cuda_pt) 


        # Extract per-residue representations
        protein_embeds = self.protein_encoder(batch_tokens, repr_layers=[33], return_contacts=True)["representations"][33].to(batch_tokens.device)

        # input llama is of shape [B, len, 5120]
        if protein_embeds.dtype != self.glm_llama_proj.weight.dtype:
            protein_embeds = protein_embeds.to(self.glm_llama_proj.weight.dtype)

        inputs_llama = self.glm_llama_proj(protein_embeds.squeeze(dim=2))#.to(protein_embeds.device)
        # atts_llama is of shape [B, len]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long)#.to(protein_embeds.device)
        batch_tokens.detach().cpu()
        protein_embeds.detach().cpu()

        return inputs_llama, atts_llama

    ##########
    # JE
    ##########
    def encode_microorganism(self, dnas, dna_labels, mode='mean'):
        #TODO
        # It is currently code for batch size 1 
        batch_dnas, batch_labels = [], []
        i, max_len, stride = 0, 512, 256



        for dna , label in zip(dnas, dna_labels):
            batch_labels.append(int(label))
        self.microorganism_encoder=self.microorganism_encoder.to(cuda_mo)
        self.glm_llama_proj_2 = self.glm_llama_proj_2.to(cuda_mo) 


        for dna, label in zip(dnas, batch_labels):
            microorganism_embeds = []
            batch_dnas = []
            while i < len(dna):
                chunk = dna[i : i+max_len] 
                batch_dnas.append(chunk)
                i += (max_len - stride)
            # model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            for batch_dna in batch_dnas:
                labels , batch_inputs, attention_masks= self.microorganism_tokenizer(batch_dna, labels=[label])#["input_ids"]
                batch_inputs = batch_inputs.to(cuda_mo) #.to(torch.cuda.current_device())
                labels = torch.tensor(labels).to(cuda_mo)#.to(torch.cuda.current_device())
                attention_masks = attention_masks.to(cuda_mo)#.to(torch.cuda.current_device())

                microorganism_embed = self.microorganism_encoder(input_ids=batch_inputs, attention_mask=attention_masks, labels=labels).hidden_states  # [1, sequence_length, 768]
                microorganism_embed = torch.mean(microorganism_embed, dim=1)
                if len(microorganism_embeds):
                    microorganism_embeds = torch.cat([microorganism_embeds, microorganism_embed], dim=0)
                else:
                    microorganism_embeds = microorganism_embed
                batch_inputs.detach().cpu()
                labels.detach().cpu()
                attention_masks.detach().cpu()

            microorganism_embeds = torch.tensor(microorganism_embeds).unsqueeze(0)
            # embedding with mean pooling
            #if mode == 'mean': microorganism_embeds = torch.mean(microorganism_embeds, dim=1)
            #else: microorganism_embeds = torch.max(microorganism_embeds[0], dim=0)[0]
            if microorganism_embeds.dtype != self.glm_llama_proj_2.weight.dtype:
                microorganism_embeds = microorganism_embeds.to(self.glm_llama_proj_2.weight.dtype)
            
            inputs_llama = self.glm_llama_proj_2(microorganism_embeds)#.to(microorganism_embeds.device) # squueze num?

            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long)#.to(microorganism_embeds.device)
        microorganism_embeds.detach().cpu()
        # this is only coded for batch 1 training
        return inputs_llama, atts_llama
    
    def llama_embed_tokens(self, *args):
        """
        Without LoRA: llama_model.model.embed_tokens
        With LoRA: llama_model.base_model.model.model.embed_tokens
        """
        if self.freeze_llama:
            return self.llama_model.model.embed_tokens(*args)
        return self.llama_model.base_model.model.model.embed_tokens(*args)
        
    #protein_embeds, microorganism_embeds, atts,mo_atts,  samples["prompt"])
    def prompt_list_wrap(self, img_embeds, img_embeds_2, atts_1, atts_2, prompt):
        if prompt:
            ########
            # JE
            #######
            img_embeds, img_embeds_2, atts_1, atts_2 = img_embeds.to(cuda_llama), img_embeds_2.to(cuda_llama), atts_1.to(cuda_llama), atts_2.to(cuda_llama)
            p_before_lst = []
            ##########
            # JE
            ##########
            p_middle_lst = []
            p_after_lst = []
            for p in prompt:
                ##########
                # JE
                ##########
                #f"###Human: <protein><proteinHere></protein>, <microorganism><microorganismHere></microorganism>, {random.choice(questions)} ###Assistant:" #JE-Q 
                p_before, next_p = p.split('<proteinHere>')[0], p.split('<proteinHere>')[-1]
                p_middle, next_p = next_p.split('<microorganismHere>')[0],  next_p.split('<microorganismHere>')[-1]
                p_after = next_p
                #p_before, p_middle, p_after = p.split(',') #<split> ## QQQ
                p_before_lst.append(p_before)
                p_middle_lst.append(p_middle)
                p_after_lst.append(p_after)

            p_before_tokens_lst = self.llama_tokenizer(
                p_before_lst, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            ##########
            # JE
            ##########
            p_middle_tokens_lst = self.llama_tokenizer(
                p_middle_lst, return_tensors="pt", add_special_tokens=True, padding=True).to(img_embeds.device)
            
            p_after_tokens_lst = self.llama_tokenizer(
                p_after_lst, return_tensors="pt", add_special_tokens=True, padding=True).to(img_embeds.device)
            # p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens_lst.input_ids)
            # p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens_lst.input_ids)
            p_before_embeds = self.llama_embed_tokens(p_before_tokens_lst.input_ids)
            ##########
            # JE
            ##########
            p_middle_embeds = self.llama_embed_tokens(p_middle_tokens_lst.input_ids)
            p_after_embeds = self.llama_embed_tokens(p_after_tokens_lst.input_ids)
            ##########
            # JE
            ##########
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_middle_embeds, img_embeds_2, p_after_embeds], dim=1) #JE-Q
            atts_img = torch.cat([atts_1, atts_2], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            img_embeds.detach().cpu()
            img_embeds_2.detach().cpu()
            atts_1.detach().cpu()
            atts_2.detach().cpu()
            p_before_embeds.detach().cpu()
            p_after_embeds.detach().cpu()
            p_middle_embeds.detach().cpu()

            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        seqs = samples["seq"] # list of seq
        seqs = [sep.upper().replace('(', '').replace(')', '') for sep in seqs]

        protein_embeds, atts = self.encode_protein(seqs)
        ##########
        # JE
        ##########
        dna_seqs = samples['dna_seq']
        dna_seqs = [seq.upper() for seq in dna_seqs]
        dna_lables = samples['dna_label']
        microorganism_embeds, mo_atts = self.encode_microorganism(dna_seqs, dna_lables)

        #JE-Q -> concat and run to prompt list wrap?
        #img_embeds, atts_img = self.prompt_list_wrap(protein_embeds, atts, samples["prompt"])
        #JE-Q -> or 
        img_embeds, atts_img = self.prompt_list_wrap(protein_embeds, microorganism_embeds, atts,mo_atts,  samples["prompt"])

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(protein_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(protein_embeds.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        # bos_embeds = self.llama_model.model.model.embed_tokens(bos)
        #####
        # JE
        #####
        bos = bos.to(cuda_llama)
        to_regress_tokens.input_ids = to_regress_tokens.input_ids.to(cuda_llama)
        atts_img = atts_img.to(cuda_llama)


        bos_embeds = self.llama_embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        #to_regress_embeds = self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        to_regress_embeds = self.llama_embed_tokens(to_regress_tokens.input_ids)
        ######
        # JE
        #####
        to_regress_tokens.attention_mask = to_regress_tokens.attention_mask.to(cuda_llama)

        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        logits = outputs.logits
        logits = torch.argmax(logits, dim=2)
        loss = outputs.loss
        to_regress_tokens.input_ids.detach().cpu()
        to_regress_tokens.attention_mask.detach().cpu()
        bos.detach().cpu()
        atts_img.detach().cpu()
        empty_targets.detach().cpu()
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
            embedding_agg = embedding_agg, 
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        stage1_ckpt = cfg.get("stage1_ckpt", "")  # load weights of encoder and LP
        if stage1_ckpt:
            print("Load GLM and LP Checkpoint: {}".format(stage1_ckpt))
            ckpt = torch.load(stage1_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        
        peft_ckpt = cfg.get("peft_ckpt", "")  # load weights of LoRA
        if peft_ckpt:
            print("Load LoRA Checkpoint: {}".format(peft_ckpt))
            ckpt = torch.load(peft_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            
        return model
