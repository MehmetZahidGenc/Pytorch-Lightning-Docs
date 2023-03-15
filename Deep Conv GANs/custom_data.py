from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomData(Dataset):
    def __init__(self, data_dir, image_size, num_channel):
        self.data_dir = data_dir
        self.image_size = image_size
        self.nc = num_channel

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(self.nc)], [0.5 for _ in range(self.nc)]),
        ])

        self.images = os.listdir(data_dir)

        self.images_len = len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item % self.images_len]

        img_path = os.path.join(self.data_dir, img)

        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)

        return img