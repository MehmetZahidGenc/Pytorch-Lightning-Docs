from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class CustomData(Dataset):
    def __init__(self, target_root_dir, input_root_dir):
        super(CustomData, self).__init__()

        self.target_root_dir = target_root_dir
        self.input_root_dir = input_root_dir

        self.list_files_target = os.listdir(self.target_root_dir)
        self.list_files_input = os.listdir(self.input_root_dir)

        # Be careful about your data input /or target image channels
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.list_files_target)

    def __getitem__(self, item):
        """ Target Image """
        target_image_file = self.list_files_target[item]
        target_image_path = os.path.join(self.target_root_dir, target_image_file)
        target_image = np.array(Image.open(target_image_path))
        target_image = Image.fromarray(target_image)
        target_image = self.transform(target_image)

        """ Input Image """
        input_image_file = self.list_files_input[item]
        input_image_path = os.path.join(self.input_root_dir, input_image_file)
        input_image = np.array(Image.open(input_image_path))
        input_image = Image.fromarray(input_image)
        input_image = self.transform(input_image)

        return input_image, target_image