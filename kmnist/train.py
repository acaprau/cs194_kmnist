import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import os
import argparse

from PIL import Image
from torch.autograd import Variable
from utils import load_train_data, load_test_data, load
from model import *


IMG_HEIGHT = 28
IMG_WIDTH = 28
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=2)
    return parser


class KMnistDataset(torch.utils.data.Dataset):
    def __init__(self, root='.', train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data, self.train_labels = load_train_data(self.root)
            self.train_labels = torch.LongTensor(self.train_labels)
        else:
            self.test_data, self.test_labels = load_test_data(self.root)
            self.test_labels =  torch.LongTensor(self.test_labels)
            
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    
def main():
    parser = _setup_parser()
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], 
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = KMnistDataset(root='./dataset',
                                  train=True,
                                  transform=transform)
    test_dataset = KMnistDataset(root='./dataset',
                                 train=False,
                                 transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True, 
                                               num_workers=args.num_workers, 
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False, 
                                              num_workers=args.num_workers, 
                                              pin_memory=True)

    model = CNNSimple()
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    model.to(device=DEVICE)
    epochs = args.epochs
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        train_correct, train_total = 0.0, 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()

            with torch.no_grad():
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                print(f'\rTraining {100 * i / len(train_loader):.2f}%, Accuracy: {train_correct / train_total:.3f}', end='')

        torch.save({'net': model.state_dict()}, 
                    f'models/{args.model}_{args.trial}.pt')
        print()

        val_correct, val_total = 0.0, 0.0
        model.eval()
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
        print(f'Validation accuracy: {val_correct / val_total:.3f}')
        model.train()

    print(f'Done!')


if __name__ == '__main__':
   main()
