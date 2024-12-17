import os
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TVF
from torch.utils.data import Dataset
import random
import torch.nn.functional as F

class Fives(Dataset):
    def __init__(self, data_path, crop_size=100, max_samples=-1, normalize=True, five_crop=True, rescale=1, augment=False):
        # set: "train" or "val"
        img_dir = os.path.join(data_path, "images")
        label_dir = os.path.join(data_path, "labels")
        self.crop_size = crop_size
        self.augment = augment
        self.five_crop = five_crop

        self.img_names = list(sorted(os.listdir(img_dir)))
        self.label_names = list(sorted(os.listdir(label_dir)))

        if max_samples > 0:
            self.img_names = self.img_names[:max_samples]
            self.label_names = self.label_names[:max_samples]

        
        transform_list = [v2.ToDtype(torch.float32, scale=True)]
        target_transform_list = [v2.ToDtype(torch.float32, scale=False)]

        if normalize:
            transform_list.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        if five_crop:
            transform_list.append(v2.FiveCrop(crop_size))
            target_transform_list.append(v2.FiveCrop(crop_size))

        self.transforms = v2.Compose(transform_list)
        self.target_transforms = v2.Compose(target_transform_list)
        
        self.images = []
        self.labels = []
        for img_name, label_name in zip(self.img_names, self.label_names):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, label_name)
            
            image = torchvision.io.read_image(img_path)
            label = (torchvision.io.read_image(label_path, mode=torchvision.io.ImageReadMode.GRAY) > 127)*1

            label = self.preprocess_label(label)
            
            h, w = image.shape[-2:]

            if rescale != 1 and (h//(rescale+1) < crop_size or w//(rescale+1) < crop_size):
                continue

            image = TVF.resize(image, min(h, w) // rescale, antialias=True, interpolation=TVF.InterpolationMode.BICUBIC)
            label = TVF.resize(label, min(h, w) // rescale, antialias=True, interpolation=TVF.InterpolationMode.BICUBIC)

            # process the labels again after resizing
            label = self.preprocess_label(label)

            self.images.append(image)
            self.labels.append(torch.stack((1-label, label), dim=0).squeeze(dim=1))



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        im_h, im_w = image.shape[-2:]

        if self.augment:
            crop_size = self.crop_size
            if self.five_crop:
                crop_size *= 2
            #i, j, h, w = v2.RandomCrop.get_params(
            #    image, output_size=(min(crop_size, im_h), min(crop_size, im_w)))
            #image = TVF.crop(image, i, j, h, w)
            #label = TVF.crop(label, i, j, h, w)
            image = TVF.center_crop(image, self.crop_size)
            label = TVF.center_crop(label, self.crop_size)            
            
            #rotate
            angle = random.choice([0, 90, 180, 270])
            image = TVF.rotate(image, angle)
            label = TVF.rotate(label, angle)
        elif not self.five_crop:
            image = TVF.center_crop(image, self.crop_size)
            label = TVF.center_crop(label, self.crop_size)

        image = self.transforms(image)
        label = self.target_transforms(label)
        
        # because of fivecrop
        if self.five_crop:
            image = torch.stack(image, dim=0)
            label = torch.stack(label, dim=0)
        return {"img": image, "seg": label}
    

    def preprocess_label(self, label):
        # Define a 3x3 kernel to check the surrounding pixels
        kernel = torch.tensor([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Apply the kernel to the label image
        neighbor_count = F.conv2d(label.float(), kernel, padding=1)
        
        # Identify isolated background pixels (value 0) surrounded by 8 foreground pixels (value 8)
        isolated_background = (label == 0) & (neighbor_count == 4)
        
        # Change isolated background pixels to foreground pixels
        label[isolated_background] = 1
        
        return label