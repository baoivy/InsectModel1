import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer, callbacks
from data.text_image_dm import TextImageDataModule, IP102Dataloader, k_shot_dataloader
from models import InsectTrainCL
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--folder', type=str, required=True, help='directory of your training folder')
    parser.add_argument('--batch_size', type=int, help='size of the batch')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
    parser.add_argument('--image_size', type=int, default=224, help='size of the images')
    parser.add_argument('--resize_ratio', type=float, default=0.75, help='minimum size of images during random crop')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to use shuffling during sampling')
    parser.add_argument('--max_epochs', type=int, default=32, help='epochs')
    parser.add_argument('--precision', type=str, default='16-mixed', help='precision')
    parser.add_argument('--embedd_dim', type=int, default=512, help='number of dimensions of the embeddings')
    parser.add_argument('--custom_tokenizer', type=int, default=None, help='tokenizer')
    parser.add_argument('--lora', type=bool, default=True, help='LoRA or not')
    parser.add_argument('--momentum', type=float, default=0.995, help='momentum')
    parser.add_argument('--queue_size', type=int, default=65536, help='momentum')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha')
    parser.add_argument('--ratio', type=float, default=0.4, help='alpha')
    parser.add_argument('--num_head', type=int, default=8, help='alpha')
    parser.add_argument('--out_dim', type=int, default=1024, help='alpha')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='alpha')
    parser.add_argument('--temp', type=float, default=4.0, help='alpha')
    parser.add_argument('--lr', type=float, default=3e-2, help='lr')
    parser.add_argument('--full_train', type=bool, default=False, help='full', required=True)
    return parser

def main(hparams):
    pl.seed_everything(76)
    wandb_logger = WandbLogger(name='Adam-32-0.001',project='InsectModel')
    lr_callback = callbacks.LearningRateMonitor(logging_interval="step")
    gpu_callback = callbacks.DeviceStatsMonitor()
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor='val/acc', 
        mode="max", 
        save_top_k=1, 
        save_last=False,
    )
    
    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    insect_names = []

    with open('classes.txt', "r") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                insect_names.append(parts[1])
    
    dm_train = k_shot_dataloader(hparams.folder, hparams.batch_size, 15)
    dm_val = IP102Dataloader(hparams.folder, hparams.batch_size, True, 'val')
    model = InsectTrainCL(hparams, insect_names)
    del hparams.model_name
    trainer = Trainer(
                    callbacks=[lr_callback, checkpoint_callback, gpu_callback], 
                    precision=hparams.precision,
                    max_epochs=hparams.max_epochs,
                    accelerator="gpu",
                    strategy="ddp",
                    logger= wandb_logger)
                    #mini_batch_size=hparams.minibatch_size)
    trainer.fit(model=model, train_dataloaders=dm_train, val_dataloaders=dm_val)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='lora_clip' ,required=True)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
