import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import yaml
import copy
from torch import optim
from .model import TextEncoder
from .model import Adapter
from loratorch.layers import MultiheadAttention as LoRA_MultiheadAttention
from torch.optim.lr_scheduler import _LRScheduler
import open_clip
import itertools
from torch.autograd import Variable
import timm

def soft_beta_loss(outputs, labels, beta, outputs_orig, num_classes=10):

    softmaxes = F.softmax(outputs, dim=1)
    n, num_classes = softmaxes.shape
    tensor_labels = Variable(torch.zeros(n, num_classes).cuda().scatter_(1, labels.long().view(-1, 1).data, 1))

    # sort outputs and labels based on confidence/entropy        
    softmaxes_orig = F.softmax(outputs_orig, dim=1)
    maximum, _ = (softmaxes_orig*tensor_labels).max(dim=1)
    maxes, indices = maximum.sort()

    sorted_softmax, sorted_labels = softmaxes[indices], tensor_labels[indices]
    sorted_softmax_orig = softmaxes_orig[indices]
    
    # generate beta labels  
    random_beta = np.random.beta(beta, 1, n)
    random_beta.sort()
    random_beta = torch.from_numpy(random_beta).cuda()
    
    # create beta smoothing labels 
    uniform = (1 - random_beta) / (num_classes - 1)
    random_beta -= uniform
    random_beta = random_beta.view(-1, 1).repeat(1, num_classes).float()
    beta_label = sorted_labels*random_beta
    beta_label += uniform.view(-1, 1).repeat(1, num_classes).float()
    
    # compute NLL loss
    loss = -beta_label * torch.log(sorted_softmax + 10**(-8))
    loss = loss.sum() / n

    return loss
    
class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)

class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]

class InsectTrainCL(pl.LightningModule):
    def __init__(self, args, classname) -> None:
        super(InsectTrainCL, self).__init__()
        self.save_hyperparameters()
        self.args = args
        
        self.momentum = args.momentum
        self.queue_size = args.queue_size
        self.minibatch_size = args.minibatch_size
        self.lr = args.lr
        self.alpha = args.alpha
        self.global_step1 = 0
        #self.isViT = 'ViT' in self.model_name
        self.ratio = args.ratio
        self.classname = classname
    
        self.open_clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        for param in self.open_clip_model.parameters():
            param.requires_grad = False

        self.text_encoder = TextEncoder(self.classname, self.open_clip_model)
        self.logit_scale = self.open_clip_model.logit_scale

        self.image_adapter = Adapter(args.num_head, args.embedd_dim, args.out_dim, args.drop_rate)
        self.image_adapter_t = Adapter(args.num_head, args.embedd_dim, args.out_dim, args.drop_rate)
        self.model_pairs = [(self.image_adapter, self.image_adapter_t)]

        self.loss_fn = nn.CrossEntropyLoss()

        self.copy_weight()


    @property
    def calculate_total_steps(self) -> int:
        return self.args.max_epochs * len(self.train_dataloader())
    
    def configure_optimizers(self):
        lr = self.lr
        self.weight_decay = 0.0000000
        parameters = [
            {
                "params": itertools.chain(
                    self.image_adapter.parameters(),
                ),
                "lr": lr,
                "weight_decay": self.weight_decay,
            },
        ]
        #optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        optimizer = optim.SGD(parameters, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_epochs)
        scheduler = ConstantWarmupScheduler(
                optimizer, lr_scheduler, 1,
                3e-2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss",
        }
        
    def training_step(self, train_batch, batch_idx):
        image_input, label = train_batch
        image_features = self.open_clip_model.encode_image(image_input)
        text_features = self.text_encoder()

        image_output = self.image_adapter(main_f=image_features)
        image_output = self.ratio*image_features + (1-self.ratio)*image_output
        
        image_output = F.normalize(image_output, p=2, dim=-1)
        language_output = F.normalize(text_features, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_output @ language_output.t()

        ce_loss = self.loss_fn(logits, label)
        with torch.no_grad():
            self.global_step1 += 1
            momentum_now = min(1 - 1 / (self.global_step1 + 1), self.momentum)
            self.momentum_update(momentum_now)
            image_output_t = self.image_adapter_t(main_f=image_features)
            image_output_t = self.ratio*image_features + (1-self.ratio)*image_output_t
            image_output_t = F.normalize(image_output_t, p=2, dim=-1)
            logits_t = logit_scale * image_output_t @ language_output.t()
            #soft_labels = F.softmax(logits_t / self.args.temp, dim = 1)
            soft_loss = soft_beta_loss(logits, label, outputs_orig=logits_t, num_classes = len(self.classname), beta=3.0)
            acc = (logits.argmax(1) == label).float().mean()
        train_loss = (1- self.alpha)*ce_loss + self.alpha*soft_loss

        self.log("train/loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss
    
    def validation_step(self, valid_batch, batch_idx):
        image_input, label = valid_batch
        image_features = self.open_clip_model.encode_image(image_input)
        text_features = self.text_encoder()

        image_output = self.image_adapter(main_f=image_features)
        image_output = self.ratio*image_features + (1-self.ratio)*image_output
        image_output = F.normalize(image_output, p=2, dim=-1)
        language_output = F.normalize(text_features, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_output @ language_output.t()

        val_loss = self.loss_fn(logits, label)

        valid_acc = (logits.argmax(1) == label).float().mean()

        self.log("val/loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", valid_acc, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss
    
    def predict_step(self, predict_batch, batch_idx):
        image_input, label = predict_batch
        image_features = self.open_clip_model.encode_image(image_input)
        text_features = self.text_encoder()

        image_output = self.image_adapter(main_f=image_features)
        image_output = self.ratio*image_features + (1-self.ratio)*image_output
        image_output = F.normalize(image_output, p=2, dim=-1)
        language_output = F.normalize(text_features, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_output @ language_output.t()
        test_acc = (logits.argmax(1) == label).float().mean()
        return test_acc.item()
    
    '''
    Adapted from CoCo code
    '''
    @torch.no_grad()
    def momentum_update(self, momentum_now):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * momentum_now + param.data * (1. - momentum_now)
    
    @torch.no_grad()
    def copy_weight(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  
                
        