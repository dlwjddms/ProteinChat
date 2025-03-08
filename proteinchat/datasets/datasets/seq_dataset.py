import os
import sys
from proteinchat.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate
import json
from torch.nn.utils.rnn import pad_sequence 
import torch
import random

#######
# JE
#######
questions = ["Can this peptide can kill this microorganism? Reply only with Yes or No.",
             "Is this peptide capable of killing this microorganism? Please answer only with 'Yes' or 'No'.",
             "Does this peptide effectively eliminate this microbe? Respond solely with 'Yes' or 'No'.",
             "Can this peptide destroy this microorganism? Provide only 'Yes' or 'No' as the answer.",
             "Is this peptide lethal to this microbe? Give your response using only 'Yes' or 'No'.",
             "Would this peptide eradicate this microorganism? Reply strictly with 'Yes' or 'No'."]
# QQQ Do you think I should make fake data ..? Like Peptide - micro - False?
# q_map = {
#     "Can this protein bind to RNA?":
#     " Reply only with Yes or No.",
#     "Can this protein bind to DNA?":
#     " Reply only with Yes or No.",
#     "What type of enzyme is this?":
#     " Choose one from Transferase, Hydrolase, Oxidoreductase, Ligase, Lyase, Isomerase, and Translocase.",
#     "What type of protein is this?":
#     " Choose one from Ribonucleoprotein and Chaperone protein",
#     "What electron acceptor or cofactor does this enzyme use?":
#     " Choose one from NAD and NADP.",
#     "What ligand can this protein bind to?":
#     " Choose one from Nucleotide, Magnesium, Zinc, Iron, S-adenosyl-L-methionine, and Manganese.",
#     "Which cellular or extracellular component can this protein be found in?":
#     " Choose one from Cytoplasm, Membrane, Nucleus, Secreted, Mitochondrion, and Plastid",
#     "What biological process does this protein involved in?":
#     " Choose one from Molecule Transport, Transcription from DNA to mRNA, Amino-acid biosynthesis, Protein biosynthesis from mRNA molecules, Lipid metabolism, tRNA processing, DNA damage, and Cell cycle."
# }
# class SeqDataset(BaseDataset):
#     def __init__(self, kw_path, text_rule_path, text_manual_path, seq_path):
#         """
#         protein (string): Root directory of protein (e.g. coco/images/)
#         ann_root (string): directory to store the annotation file
#         """
#         # print("______Enter Seq Dataset____")
#         # super().__init__(vis_processor, text_processor)
#         # self.qa_path = qa_path
#         # self.seq_path = seq_path

#         self.kw = json.load(open(kw_path, "r")) 
#         self.rule = json.load(open(text_rule_path, "r"))
#         self.manual = json.load(open(text_manual_path, "r"))
#         self.sequence = json.load(open(seq_path, "r"))

#         self.rate = {'kw':1, 'rule':1, 'manual':4}
#         self.len_kw = len(self.kw)
#         self.len_rule = len(self.rule)
#         self.len_manual = len(self.manual)

#         self.split1 = self.rate['kw'] * self.len_kw 
#         self.split2 = self.split1 + self.rate['rule'] * self.len_rule
#         self.split3 = self.split2 + self.rate['manual'] * self.len_manual 

#     def __len__(self):
#         return self.split3

#     def __getitem__(self, index):
        
#         if index < self.split1: # sample kw 
#             uniprot_id = self.kw[index]["uniprot_id"]
#             answer = self.kw[index]["A"]
#             query = self.kw[index]['Q']
#             query += q_map[query]
#             prompt = f"###Human: <protein><proteinHere></protein> {query} ###Assistant:"
#         elif index < self.split2: # sample rule based functionality
#             true_index  = (index - self.split1) % self.len_rule
#             uniprot_id = self.rule[true_index]["uniprot_id"]
#             answer = self.rule[true_index]["caption"]
#             prompt = f"###Human: <protein><proteinHere></protein> {random.choice(questions)} ###Assistant:"
#         else: # sample manual annotated functionality
#             true_index  = (index - self.split2) % self.len_manual
#             uniprot_id = self.manual[true_index]["uniprot_id"]
#             answer = self.manual[true_index]["caption"]
#             prompt = f"###Human: <protein><proteinHere></protein> {random.choice(questions)} ###Assistant:"
        
#         seq = self.sequence[uniprot_id]

#         if len(seq) > 600:
#             seq = seq[:600]

#         return {
#             "seq": seq,
#             "text_input": answer,
#             "prompt": prompt
#         }

#     # stage1-Qformer
#         # uniprot_id = self.annotation[index]["uniprot_id"]
#         # seq = self.sequence[uniprot_id]
#         # answer = self.annotation[index]["name"]

#         # if len(seq) > 1024:
#         #     seq = seq[:1024]

#         # return {
#         #     "seq": seq,
#         #     "text_input": answer
#         # }


class SeqDataset(BaseDataset):
    def __init__(self, data_path):
        """
        protein (string): Root directory of protein (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # print("______Enter Seq Dataset____")
        # super().__init__(vis_processor, text_processor)
        # self.qa_path = qa_path
        # self.seq_path = seq_path

        self.data = json.load(open(data_path, "r"))
        
        self.length = len(self.data)
        self.classes = {}
        self.class_count = 0
        for item in self.data: 
            if item['microorganism'] in self.classes: continue
            self.classes[item['microorganism']] = self.class_count
            self.class_count +=1
        
        random.shuffle(self.data)
        print('Size of train data', len(self.data))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seq = self.data[index]["sequence"]
        #######
        # JE
        #######
        dna_seq = self.data[index]["micoorganism_sequence"]
        dna_label = self.classes[self.data[index]['microorganism']]
        function_text = self.data[index]["Label"] #answer 
        # prompt = f"###Human: <protein><proteinHere></protein> <structure><structureHere></structure> {random.choice(questions)} ###Assistant:"
                
        #######
        # JE
        #######
        prompt = f"###Human: <protein><proteinHere></protein> <microorganism><microorganismHere></microorganism> {random.choice(questions)} ###Assistant:" #JE-Q 
        #return (seq, dna_seq, dna_label, function_text, prompt) #NEW_JE
        return {
            "seq": seq,
            "dna_seq": dna_seq,
            "dna_label": dna_label,
            "text_input": function_text,#QQQ: WHat should be the answer ..? : Yes/No ? 
            "prompt": prompt
        }

    # stage1-Qformer
        # uniprot_id = self.annotation[index]["uniprot_id"]
        # seq = self.sequence[uniprot_id]
        # answer = self.annotation[index]["name"]

        # if len(seq) > 1024:
        #     seq = seq[:1024]

        # return {
        #     "seq": seq,
        #     "text_input": answer
        # }



