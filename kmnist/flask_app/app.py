import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import argparse

from PIL import Image
from torch.autograd import Variable


import os
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory, url_for

__author__ = 'ibininja'
IMG_HEIGHT = 28
IMG_WIDTH = 28

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

class CNNConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(CNNConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out += identity
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        return out


class CNNSimple(nn.Module):
    def __init__(self):
        super(CNNSimple, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32*4*4, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = flatten(out)
        out = self.dropout(out)
        out = self.fc1(out)

        return out


class CNNDeeper(nn.Module):
    def __init__(self):
        super(CNNDeeper, self).__init__()
        IM_HEIGHT = 28
        IM_WIDTH = 28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block1 = CNNConvBlock(in_channels=64, out_channels=64)
        self.drop1 = nn.Dropout(p=0.5)
        self.block2 = CNNConvBlock(in_channels=64, out_channels=64)
        self.drop2 = nn.Dropout(p=0.5)
        self.block3 = CNNConvBlock(in_channels=64, out_channels=128)
        self.drop3 = nn.Dropout(p=0.5)
        self.block4 = CNNConvBlock(in_channels=128, out_channels=128)
        self.drop4 = nn.Dropout(p=0.5)
        self.block5 = CNNConvBlock(in_channels=128, out_channels=128)
        self.drop5 = nn.Dropout(p=0.5)
        self.block6 = CNNConvBlock(in_channels=128, out_channels=128)
        self.drop6 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        out_dim = IM_HEIGHT - 2 * 6
        self.fc1 = nn.Linear(128*out_dim*out_dim, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.block1(out)
        out = self.drop1(out)
        out = self.block2(out)
        out = self.drop2(out)
        out = self.block3(out)
        out = self.drop3(out)
        out = self.block4(out)
        out = self.drop4(out)
        out = self.block5(out)
        out = self.drop5(out)
        out = self.block6(out)
        out = self.drop6(out)
        out = flatten(out)
        out = self.fc1(out)

        return out

def load(f):
    return np.load(f)['arr_0']

def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

def write_numpy_image(img, filepath='img.png'):
    img = Image.fromarray(img, mode='L')
    img.save(filepath)

def write_pillow_image(img, filepath='img.png'):
    img.save(filepath)

def read_image_to_pillow(filepath):
    img = Image.open(filepath)
    return img

def read_pillow_from_dataset(dataset='test', element=0):
    if dataset == 'test':
        data = load('dataset/kmnist-test-imgs.npz')
    else:
        data = load('dataset/kmnist-train-imgs.npz')
    img = data[element]
    img = Image.fromarray(img, mode='L')
    return img

def main(path):
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    LABELS_10 =["お", "き", "す", "つ", "な", "は", "ま", "や", "れ", "を"]
    LABELS_49 = ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち",
                "つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め"
                "も","や","ゆ","よ","ら","り","る","れ","ろ","わ","ゐ","ゑ","を","ん","ゝ"]
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    inv_transform = transforms.Compose([transforms.Normalize((-0.4242,), (3.2457,)), 
                                        transforms.ToPILImage()])

    path = 'kmnist_test/' + path
    img = read_image_to_pillow(path)
    #print(f'Inputted image shape: {img.shape}')
    #if len(img.shape) == 3 and img.shape[0] > 1:
    #    img = img[0]
    #img = Image.fromarray(img, mode='L')
    img = transform(img)
    img = img.unsqueeze(0)
    img.to(device=DEVICE)
    
    if False:
        state = torch.load('models/simple_0.pt')
        model = CNNSimple()
        model.load_state_dict(state['net'])
    elif True:
        state = torch.load('models/deeper_0.pt', map_location=torch.device('cpu'))
        model = CNNDeeper()
        model.load_state_dict(state['net'])
    else:
        print(f'not implemented')
        return

    model.eval()
    model.to(device=DEVICE)
    with torch.no_grad():
        pred = model(img)
        pred = pred.squeeze()
    _, pred = pred.max(0)
    print(f"I think you've given me {LABELS_10[pred]}!")
    print(f'How did I do?')
    
    return f"I think you've given me {LABELS_10[pred]}! This corresponds to label {pred}." + """<form action="/submit" method="post">
    <button name="back" type="submit">go back</button>
</form>"""

def redirect_url(default='index'):
    return url_for(default)

def some_view():
    # some action
    return redirect(redirect_url())

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/submit")
def submit():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
        print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

    # return send_from_directory("images", filename, as_attachment=True)
    #return render_template("index.html", image_name=filename)
    return main(filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
