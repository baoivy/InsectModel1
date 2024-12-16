import torch
import torch.nn as nn
from loratorch.layers import MultiheadAttention as LoRA_MultiheadAttention
from timm.models.vision_transformer import VisionTransformer
import copy
from typing import List, Tuple
from torch import Tensor
import math
import clip

def create_child_from_parent(parent_instance, child_class, **child_args):
    child_instance = child_class.__new__(child_class)
    for attr, value in vars(parent_instance).items():
        if hasattr(child_instance, attr):
            setattr(child_instance, attr, copy.deepcopy(value))
    child_class.__init__(child_instance, **child_args)
    return child_instance

def add_lora_layer_to_open_clip(open_clip_model, r: int = 4, num_classes: int = 0, lora_layer=None):
    if num_classes != 768:
        raise ValueError(
            "num_classes should be 768 for OpenCLIP, may need to implement a new head for other num_classes")

    vit_model = open_clip_model.visual

    for param in vit_model.parameters():
        param.requires_grad = False

    assert r > 0
    if lora_layer is not None:
        lora_layer = lora_layer
    else:
        lora_layer = list(range(len(vit_model.transformer.resblocks)))
        block_list = enumerate(vit_model.transformer.resblocks)

    for param in vit_model.parameters():
        param.requires_grad = False

    for t_layer_i, blk in block_list:
        # If we only want few lora layer instead of all
        if t_layer_i not in lora_layer:
            continue
        blk.attn = create_child_from_parent(blk.attn, LoRA_MultiheadAttention, embed_dim=blk.attn.embed_dim,
                                            num_heads=blk.attn.num_heads, enable_lora=['q', 'k', 'v'], r=r)
    open_clip_model.visual = vit_model

    # Do the same for the language model
    language_model = open_clip_model.transformer
    for param in language_model.parameters():
        param.requires_grad = False

    assert r > 0
    if lora_layer is not None:
        lora_layer = lora_layer
    else:
        lora_layer = list(range(len(language_model.resblocks)))
        block_list = enumerate(language_model.resblocks)

    for param in language_model.parameters():
        param.requires_grad = False

    for t_layer_i, blk in block_list:
        # If we only want few lora layer instead of all
        if t_layer_i not in lora_layer:
            continue
        blk.attn = create_child_from_parent(blk.attn, LoRA_MultiheadAttention, embed_dim=blk.attn.embed_dim,
                                            num_heads=blk.attn.num_heads, enable_lora=['q', 'k', 'v'], r=r)
    open_clip_model.transformer = language_model

    return open_clip_model

# MODIFIED FROM https://github.com/JamesQFreeman/LoRA-ViT/blob/main/lora.py
class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: VisionTransformer, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        # dim = vit_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.reset_classifier(num_classes=num_classes)
            # self.lora_vit.head = nn.Linear(
            #     self.dim, num_classes)

    def reset_classifier(self, num_classes):
        self.lora_vit.reset_classifier(num_classes=num_classes)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


class LoRA_ViT_OpenCLIP(nn.Module):
    def __init__(self, vit_model, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT_OpenCLIP, self).__init__()
        if num_classes != 768:
            raise ValueError(
                "num_classes should be 768 for OpenCLIP, may need to implement a new head for other num_classes")
        assert r > 0
        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.resblocks)))
            block_list = enumerate(vit_model.transformer.resblocks)

        # dim = vit_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in block_list:
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            blk.attn = create_child_from_parent(blk.attn, LoRA_MultiheadAttention, embed_dim=blk.attn.embed_dim, num_heads=blk.attn.num_heads, enable_lora=['q', 'k', 'v'], r=r)

        self.lora_vit = vit_model

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)
    
# MODIFIED FROM https://github.com/JamesQFreeman/LoRA-barcode_bert/blob/main/lora.py

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRA_bert(nn.Module):
    def __init__(self, model, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_bert, self).__init__()

        assert r > 0
        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(model.encoder.layer)))

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in model.parameters():
            param.requires_grad = False

        for layer_idx, layer in enumerate(model.encoder.layer):
            if layer_idx not in self.lora_layer:
                continue
            w_q_linear = layer.attention.self.query
            w_v_linear = layer.attention.self.value
            dim = layer.attention.self.query.in_features

            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            layer.attention.self.query = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            layer.attention.self.value = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_bert = model

        if num_classes > 0:
            self.proj = nn.Linear(self.lora_bert.pooler.dense.out_features, num_classes)


    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x) -> Tensor:

        return self.proj(self.lora_bert(**x).last_hidden_state.mean(dim=1))

class LoRA_bert_OpenCLIP(nn.Module):
    def __init__(self, bert_model, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_bert_OpenCLIP, self).__init__()
        if num_classes != 768:
            raise ValueError(
                "num_classes should be 768 for OpenCLIP, may need to implement a new head for other num_classes")
        assert r > 0
        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(bert_model.resblocks)))
            block_list = enumerate(bert_model.resblocks)

        # lets freeze first
        for param in bert_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in block_list:
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            blk.attn = create_child_from_parent(blk.attn, LoRA_MultiheadAttention, embed_dim=blk.attn.embed_dim, num_heads=blk.attn.num_heads, enable_lora=['q', 'k', 'v'], r=r)

        self.lora_bert = bert_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x: Tensor) -> Tensor:
        x = clip.tokenize(x).to(self.device)
        print(x)
        print(self.lora_bert)

        return self.lora_bert(x)