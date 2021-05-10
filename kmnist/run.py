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
LABELS_10 =["お", "き", "す", "つ", "な", "は", "ま", "や", "れ", "を"]
LABELS_49 = ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち",
             "つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め"
             "も","や","ゆ","よ","ら","り","る","れ","ろ","わ","ゐ","ゑ","を","ん","ゝ"]

def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./dataset/kmnist-test-imgs.npz')
    parser.add_argument('--model', type=str, default='simple')
    return parser

def main():
    parser = _setup_parser()
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    inv_transform = transforms.Compose([transforms.Normalize((-0.4242,), (3.2457,)), 
                                        transforms.ToPILImage()])

    path = os.path.expanduser(args.path)
    img = load(path)
    print(f'Inputted image shape: {img.shape}')
    if len(img.shape) == 3 and img.shape[0] > 1:
        img = img[0]
    img = Image.fromarray(img, mode='L')
    img = transform(img)
    img = img.unsqueeze(0)
    img.to(device=DEVICE)
    
    if args.model == 'simple':
        state = torch.load('models/simple_0.pt')
        model = CNNSimple()
        model.load_state_dict(state['net'])
    elif args.model == 'deeper':
        state = torch.load('models/deeper_0.pt')
        model = CNNDeeper()
        model.load_state_dict(state['net'])
    else:
        print(f'{args.model} not implemented')
        return

    model.eval()
    model.to(device=DEVICE)
    with torch.no_grad():
        pred = model(img)
        pred = pred.squeeze()
    _, pred = pred.max(0)
    print(f"I think you've given me {LABELS_10[pred]}!")
    print(f'How did I do?')


if __name__ == '__main__':
   main()
