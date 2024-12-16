import pytorch_lightning as pl
import open_clip
import torch
from data.text_image_dm import IP102Dataloader
import torchvision.transforms as transforms
from models import InsectTrainCL
import argparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from tqdm import tqdm


def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--folder', type=str, required=True, help='directory of your training folder')
    parser.add_argument('--batch_size', type=int, help='size of the batch')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
    parser.add_argument('--image_size', type=int, default=224, help='size of the images')
    parser.add_argument('--resize_ratio', type=float, default=0.75, help='minimum size of images during random crop')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to use shuffling during sampling')
    parser.add_argument('--max_epochs', type=int, default=32, help='epochs')
    parser.add_argument('--precision', type=int, default=16, help='precision')
    parser.add_argument('--embedd_dim', type=int, default=768, help='number of dimensions of the embeddings')
    return parser

def get_features(trainer, model, dataloader):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images_input, labels in tqdm(dataloader):
            image_features, _ = trainer.predict(model, images_input, None)

            all_features.append(image_features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def main():
    # Load the model
    args = add_argparse_args([])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InsectTrainCL.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=31.ckpt", map_location=device)

    model.eval()
    transform = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    
    train_dataloader = IP102Dataloader(args.folder, args.barch_size, True, 'train')
    test_dataloader = IP102Dataloader(args.folder, args.batch_size , False, 'test')
    trainer = pl.Trainer(accelerator="gpu")
    # Prepare the inputs
    predict_label = []
    
    
    # Calculate the image features
    train_features, train_labels = get_features(trainer, model, train_dataloader)
    test_features, test_labels = get_features(trainer, model, test_dataloader)

    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)

    accuracy = accuracy_score(test_labels, predict_label)
    f1 = f1_score(test_labels, predict_label, average='macro')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')


if __name__ == '__main__':
    main()