import torch.nn.functional as F
import timm
import torch.nn as nn
from utils import LoRA_ViT_timm, LoRA_ViT_OpenCLIP, LoRA_bert, LoRA_bert_OpenCLIP, add_lora_layer_to_open_clip
import numpy as np
from typing import Optional
import torch
import clip
from transformers import AutoModel
import open_clip

class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class Adapter(nn.Module):
    def __init__(self, num_head, embedd_dim, hidden_dim, drop_rate) -> None:
        super().__init__()
        self.linear = FeedForward(embedd_dim, embedd_dim, hidden_dim)
        self.att = nn.MultiheadAttention(embedd_dim, num_head)
        self.dropout = nn.Dropout(drop_rate)
        self.norm = nn.LayerNorm(embedd_dim)

    def forward(self, main_f, mask=None):
        out = main_f
        self_att, _ = self.att(main_f, main_f, main_f, attn_mask=mask)
        out = self.norm(out + self.dropout(self_att))
        out = self.norm(out + self.dropout(self.linear(out)))
        return out

class TextEncoder(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.clip_model = clip_model
        self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip', context_length=77)
    
    def forward(self):
        temp = "a photo of a {}, a type of insect."
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([self.tokenizer(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x
    

