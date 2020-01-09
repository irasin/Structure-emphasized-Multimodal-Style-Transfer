import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import io, transform
from PIL import Image


SIZE = 512

trans = transforms.Compose([transforms.ToTensor()])


class PreprocessDataset(Dataset):
    def __init__(self, content_dir, style_dir, image_transforms=trans):
        content_dir_resized = content_dir + f'_resized_{SIZE}_{SIZE}'
        style_dir_resized = style_dir + f'_resized_{SIZE}_{SIZE}'
        if not (os.path.exists(content_dir_resized) and
                os.path.exists(style_dir_resized)):
            os.mkdir(content_dir_resized)
            os.mkdir(style_dir_resized)
            self._resize(content_dir, content_dir_resized)
            self._resize(style_dir, style_dir_resized)
        content_images = glob.glob((content_dir_resized + '/*'))
        np.random.shuffle(content_images)
        style_images = glob.glob(style_dir_resized + '/*')
        np.random.shuffle(style_images)
        self.images_pairs = list(zip(content_images, style_images))
        self.transforms = image_transforms

    @staticmethod
    def _resize(source_dir, target_dir):
        print(f'Start resizing {source_dir} ')
        for i in tqdm(os.listdir(source_dir)):
            filename = os.path.basename(i)
            try:
                image = io.imread(os.path.join(source_dir, i))
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    image = transform.resize(image, (SIZE, SIZE), mode='reflect', anti_aliasing=True)
                    io.imsave(os.path.join(target_dir, filename), image)
            except:
                continue

    def __len__(self):
        return len(self.images_pairs)

    def __getitem__(self, index):
        content_image_path, style_image_path = self.images_pairs[index]
        content_image = Image.open(content_image_path)
        style_image = Image.open(style_image_path)

        content_image_tensor = self.transforms(content_image)
        style_image_tensor = self.transforms(style_image)
        return content_image_path, style_image_path, content_image_tensor, style_image_tensor


def get_loader(dataset, batch_size, shuffle=True):
    def my_collate_fn(batch):
        content_image_path = [sample[0] for sample in batch]
        style_image_path = [sample[1] for sample in batch]
        content_image_tensor = torch.cat([sample[2].unsqueeze(dim=0) for sample in batch], dim=0)
        style_image_tensor = torch.cat([sample[3].unsqueeze(dim=0) for sample in batch], dim=0)
        return content_image_path, style_image_path, content_image_tensor, style_image_tensor

    return DataLoader(dataset, batch_size, shuffle, collate_fn=my_collate_fn)
