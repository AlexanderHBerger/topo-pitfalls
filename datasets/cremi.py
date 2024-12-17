import os
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TVF
from torch.utils.data import Dataset
import random
import torch.nn.functional as F

class Cremi(Dataset):
    def __init__(self, data_path, crop_size=100, max_samples=-1, normalize=True, five_crop=True, rescale=1, augment=False, fill_hole=False):
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
        
        if five_crop and not augment:
            transform_list.append(v2.FiveCrop(crop_size))
            target_transform_list.append(v2.FiveCrop(crop_size))

        self.transforms = v2.Compose(transform_list)
        self.target_transforms = v2.Compose(target_transform_list)
        
        self.images = []
        self.labels = []
        for img_name, label_name in zip(self.img_names, self.label_names):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, label_name)
            
            image = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.GRAY)
            label = (torchvision.io.read_image(label_path, mode=torchvision.io.ImageReadMode.GRAY) > 127)*1

            if fill_hole:
                label = self.preprocess_label(label)
            
            # h, w = image.shape[-2:]

            # if rescale != 1 and (h//(rescale+1) < crop_size or w//(rescale+1) < crop_size):
            #     continue
            
            # make quadratic and rescale
            # image = TVF.resize(image, min(h, w) // rescale, antialias=True)
            # label = TVF.resize(label, min(h, w) // rescale, antialias=True)

            im_h, im_w = image.shape[-2:]

            if not self.augment:
                # transform the image into 4 parts (for each corner) and add them to the list
                # left top corner
                image1 = TVF.crop(image, 0, 0, crop_size, crop_size)
                label1 = TVF.crop(label, 0, 0, crop_size, crop_size)
                # right top corner
                image2 = TVF.crop(image, 0, im_w - crop_size, crop_size, crop_size)
                label2 = TVF.crop(label, 0, im_w - crop_size, crop_size, crop_size)
                # left bottom corner
                image3 = TVF.crop(image, im_h - crop_size, 0, crop_size, crop_size)
                label3 = TVF.crop(label, im_h - crop_size, 0, crop_size, crop_size)
                # right bottom corner
                image4 = TVF.crop(image, im_h - crop_size, im_w - crop_size, crop_size, crop_size)
                label4 = TVF.crop(label, im_h - crop_size, im_w - crop_size, crop_size, crop_size)

                # append to list
                self.images.extend([image1, image2, image3, image4])
                # append to list as one-hot encoded tensor
                self.labels.extend([torch.stack((1-label1, label1), dim=0).squeeze(dim=1), torch.stack((1-label2, label2), dim=0).squeeze(dim=1), torch.stack((1-label3, label3), dim=0).squeeze(dim=1), torch.stack((1-label4, label4), dim=0).squeeze(dim=1)])

            else:   # QUESTION: In case of not augment (testing), the main image should not be added to the list, right? (The else statement was missing)

                self.images.append(image)
                self.labels.append(torch.stack((1-label, label), dim=0).squeeze(dim=1))


    def preprocess_label(self, label):
        # Define a 3x3 kernel to check the surrounding pixels
        kernel = torch.tensor([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Apply the kernel to the label image
        neighbor_count = F.conv2d(label.float(), kernel, padding=1)
        
        # Identify isolated background pixels (value 0) surrounded by 8 foreground pixels (value 8)
        isolated_background = (label == 0) & (neighbor_count == 8)
        
        # Change isolated background pixels to foreground pixels
        label[isolated_background] = 1
        
        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        im_h, im_w = image.shape[-2:]

        # If augment (because otherwise it has been cropped already in init) and not five_crop, do random crop
        if self.augment and not self.five_crop:
            i, j, h, w = v2.RandomCrop.get_params(
                image, output_size=(self.crop_size, self.crop_size))
            image = TVF.crop(image, i, j, h, w)
            label = TVF.crop(label, i, j, h, w)
        
        if self.augment:
            # rotate
            angle = random.choice([0, 90, 180, 270])
            image = TVF.rotate(image, angle)
            label = TVF.rotate(label, angle)

        image = self.transforms(image)
        label = self.target_transforms(label)
        
        # because of fivecrop
        if self.five_crop:
            image = torch.stack(image, dim=0)
            label = torch.stack(label, dim=0)

        return {"img": image, "seg": label}
    


class CremiFullData(Dataset):
    def __init__(self, train_path, test_path):
        # set: "train" or "val"
        img_dir_train = os.path.join(train_path, "images")
        label_dir_train = os.path.join(train_path, "labels")

        img_dir_test = os.path.join(test_path, "images")
        label_dir_test = os.path.join(test_path, "labels")

        self.train_img_names = list(sorted(os.listdir(img_dir_train)))
        self.train_label_names = list(sorted(os.listdir(label_dir_train)))

        self.test_img_names = list(sorted(os.listdir(img_dir_test)))
        self.test_label_names = list(sorted(os.listdir(label_dir_test)))


        transform_list = [v2.ToDtype(torch.float32, scale=True)]
        target_transform_list = [v2.ToDtype(torch.float32, scale=False)]


        self.transforms = v2.Compose(transform_list)
        self.target_transforms = v2.Compose(target_transform_list)
        
        self.images = []
        self.labels = []
        for img_name, label_name in zip(self.train_img_names, self.train_label_names):
            img_path = os.path.join(img_dir_train, img_name)
            label_path = os.path.join(label_dir_train, label_name)
            
            image = torchvision.io.read_image(img_path)
            label = (torchvision.io.read_image(label_path, mode=torchvision.io.ImageReadMode.GRAY) > 127)*1

            self.images.append(image)
            self.labels.append(torch.stack((1-label, label), dim=0).squeeze(dim=1))

        for img_name, label_name in zip(self.test_img_names, self.test_label_names):
            img_path = os.path.join(img_dir_test, img_name)
            label_path = os.path.join(label_dir_test, label_name)
            
            image = torchvision.io.read_image(img_path)
            label = (torchvision.io.read_image(label_path, mode=torchvision.io.ImageReadMode.GRAY) > 127)*1

            self.images.append(image)
            self.labels.append(torch.stack((1-label, label), dim=0).squeeze(dim=1))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        im_h, im_w = image.shape[-2:]

        return {"img": image, "seg": label}