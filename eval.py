import pytorch_lightning as pl
import open_clip
import torch
from data.text_image_dm import IP102Dataloader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import InsectTrainCL
import argparse
from argparse import ArgumentParser

def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--folder', type=str, help='directory of your training folder')
    parser.add_argument('--batch_size', type=int, help='size of the batch')
    parser.add_argument('--checkpoint', type=str, default = 'InsectModel/4k2vhfz8/checkpoints/epoch=11-step=312.ckpt', help='Checkpoint path')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
    parser.add_argument('--image_size', type=int, default=224, help='size of the images')
    parser.add_argument('--resize_ratio', type=float, default=0.75, help='minimum size of images during random crop')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to use shuffling during sampling')
    parser.add_argument('--max_epochs', type=int, default=32, help='epochs')
    parser.add_argument('--precision', type=int, default=16, help='precision')
    parser.add_argument('--embedd_dim', type=int, default=768, help='number of dimensions of the embeddings')
    return parser

def main():
    # Load the model
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='lora_clip')
    parser = add_argparse_args(parser)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InsectTrainCL.load_from_checkpoint(args.checkpoint)

    model.eval()
    transform = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_loader = IP102Dataloader('ip102_v1.1', 1, False, 'test')
    trainer = pl.Trainer(accelerator="gpu")

    insect_names = []

    with open('classes.txt', "r") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                insect_names.append(parts[1])

    with torch.no_grad():
        test_acc = trainer.predict(model, test_loader)

    overall_accuracy = sum(test_acc) / len(test_acc)
    print('Accuracy:',  overall_accuracy)


if __name__ == '__main__':
    main()