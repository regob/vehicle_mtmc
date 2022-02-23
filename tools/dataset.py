from torch.utils.data import Dataset
from PIL import Image
import os
import glob


class TestDataset(Dataset):
    def __init__(self, img_root, transform=None):
        self.imgs = glob.glob(img_root + "/**/*.*", recursive=True)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.imgs[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path
