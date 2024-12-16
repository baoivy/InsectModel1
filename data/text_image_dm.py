# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
from random import randint, choice

import PIL
import argparse
import clip
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
import json
import numpy as np
import torchvision.transforms as transforms
import requests
import os 
from transformers import AutoTokenizer

class IP102(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split='train'):
        assert split in ['train', 'val', 'test']
        self.image_list = []
        self.label_list = []

        with open(os.path.join(root, f'{split}.txt')) as f:
            lines = f.readlines()

        for line in lines:
            filename, class_id = line.strip().split()
            class_id = int(class_id)

            self.image_list.append(os.path.join(root, 'images', filename))
            self.label_list.append(class_id)

        self.transform = transform

        print(f"Dataset has {len(self.image_list)} images")
    
    def get_classes(self):
        return self.label_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = self.label_list[index]
        image = PIL.Image.open(image_path)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

class Insect1MDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_filepath, transform=None, shuffle=False, custom_tokenizer=None):
        self.image_list = []
        self.caption_list = []

        with open(metadata_filepath) as f:
            metadata = json.load(f)

        insect_records = metadata['insect_records']
        desc_records = metadata['description_records']
        
        desc_dict = {}
        self.descriptions = {}

        for item in desc_records:
            desc_dict[item['id']] = item

        for i, item in enumerate(insect_records):
            if i == 827:
                break
            self.image_list.append(str('insect-images/insect' + str(item['id']) + '.jpg'))
            attri = []
            for key, value in item.items():
                if key in ['No Taxon', 'image_url', 'id', 'description_ids']:
                    continue
                attri.append(value)

            caption = 'An image of ' + ', '.join(attri)
            captions = [caption.lower()]
            for desc_id in item['description_ids']:
                assert desc_id in desc_dict
                caption = desc_dict[desc_id]['description']

                sentences = caption.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) <= 5:
                        continue

                    captions.append(sentence.lower())

            self.caption_list.append(captions)

        self.transform = transform
        self.custom_tokenizer = custom_tokenizer
        if custom_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        caption = np.random.choice(self.caption_list[index])
        for _ in range(10):
            try:
                image = PIL.Image.open(image_path) 
                image = image.convert("RGB")
                break
            except:
                print(f'Could not open {image_path}')
                index = np.random.randint(0, len(self.image_list))
                image_path = self.image_list[index]
                caption = np.random.choice(self.caption_list[index])

        tokenized_text = clip.tokenize(caption, context_length=77,  truncate = True)[0] if self.custom_tokenizer is None else self.tokenizer.encode_plus(caption, return_tensors='pt', padding='max_length', max_length=128, truncation=True)['input_ids']
        if self.transform:
            image = self.transform(image)

        return image, tokenized_text

class TextImageDataModule(LightningDataModule):
    def __init__(self, args):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (transformers.AutoTokenizer, optional): The tokenizer to use on the text. Defaults to None.
        """
        super().__init__()
        self.folder = args.folder
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.image_size = args.image_size
        self.resize_ratio = args.resize_ratio
        self.custom_tokenizer = args.custom_tokenizer
        self.shuffle = args.shuffle
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    def setup(self, stage=None):
        self.dataset = Insect1MDataset(metadata_filepath=self.folder, transform=self.transform, shuffle=self.shuffle)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True , collate_fn=self.dl_collate_fn)
    
    def dl_collate_fn(self, batch):
        if self.custom_tokenizer is None:
            return torch.stack([row[0] for row in batch]), torch.stack([row[1] for row in batch])
        return torch.stack([row[0] for row in batch]), self.custom_tokenizer([row[1] for row in batch], padding=True, truncation=True, return_tensors="pt")

def IP102Dataloader(root, batch_size=8, shuffle=False, type='train'):
    transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = IP102(root=root, transform=transform, split=type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def k_shot_dataloader(root, batch_size, k):
    transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = IP102(root=root, transform=transform, split='train')
    # Group data by class
    class_indices = {i: [] for i in range(102)}
    for idx, (image, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Sample k examples per class
    sampled_indices = []
    for class_id, indices in class_indices.items():
        sampled_indices.extend(np.random.choice(indices, k, replace=False))

    # Create k-shot dataset
    sampled_data = torch.utils.data.Subset(dataset, sampled_indices)
    dataloader = DataLoader(sampled_data, batch_size=batch_size, shuffle=True)

    return dataloader