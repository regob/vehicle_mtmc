import argparse
import torch
import tqdm
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from dataset import TestDataset
import os
import shutil


parser = argparse.ArgumentParser(
    description="Run a model on a dataset of images, and sort the images into folders based on the prediction")
parser.add_argument("--input_dir", type=str,
                    help="Directory to read images from (recursively)")
parser.add_argument("--save_dir", type=str, help="Output directory")
parser.add_argument("--model", type=str, help="Model to use for prediction")
parser.add_argument("--size", type=int, default=224, help="Image input size")
parser.add_argument("--num_workers", type=int, default=0,
                    help="Number of worker threads for data loading")
parser.add_argument("--max_images", type=int, default=-1,
                    help="Limit the number of images to run inference for")
args = parser.parse_args()

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if not os.path.isfile(args.model):
    raise ValueError("Model file does not exist")
net = torch.load(args.model).to(device)
net.eval()

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])

if not os.path.isdir(args.input_dir):
    raise ValueError("Input dir does not exist")
dataset = TestDataset(args.input_dir, transform=trans)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

testloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, num_workers=args.num_workers)

for i, dat in enumerate(testloader):
    if args.max_images >= 0 and i >= args.max_images:
        break

    inputs, labels = dat
    inputs = inputs.to(device)
    outputs = net(inputs)
    filename = os.path.split(labels[0])[1]
    pred = outputs.argmax().item()

    dirout = os.path.join(args.save_dir, str(pred))
    if not os.path.exists(dirout):
        os.makedirs(dirout)
    shutil.copy2(labels[0], os.path.join(dirout, filename))


print("Prediction successful. Outputs are at: {}".format(args.save_dir))
