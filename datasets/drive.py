import os
from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TVF
from torch.utils.data import Dataset
import random
import skimage.measure
import skimage.morphology

class Drive(Dataset):
    def __init__(self, data_path, crop_size=100, max_samples=-1, five_crop=True, rescale=1, augment=False, fill_hole=False, eight_connectivity=True):
        img_dir = os.path.join(data_path, "images")
        label_dir = os.path.join(data_path, "1st_manual")
        mask_dir = os.path.join(data_path, "mask")
        self.crop_size = crop_size
        self.five_crop = five_crop
        self.augment = augment

        self.img_names = list(sorted(os.listdir(img_dir)))
        self.label_names = list(sorted(os.listdir(label_dir)))
        self.mask_names = list(sorted(os.listdir(mask_dir)))

        if max_samples > 0:
            self.img_names = self.img_names[:max_samples]
            self.label_names = self.label_names[:max_samples]
            self.mask_names = self.mask_names[:max_samples]
        
        transform_list = [v2.ToDtype(torch.float32, scale=True)]
        target_transform_list = [v2.ToDtype(torch.float32, scale=False)]
        
        if five_crop:
            transform_list.append(v2.FiveCrop(crop_size))
            target_transform_list.append(v2.FiveCrop(crop_size))

        self.transforms = v2.Compose(transform_list)
        self.target_transforms = v2.Compose(target_transform_list)
        
        self.images = []
        self.labels = []
        self.masks = []
        for img_name, label_name, mask_name in zip(self.img_names, self.label_names, self.mask_names):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, label_name)
            mask_path = os.path.join(mask_dir, mask_name)
            
            image = TVF.pil_to_tensor(Image.open(img_path))
            label = (torchvision.io.read_image(label_path, mode=torchvision.io.ImageReadMode.GRAY) > 127) * 1
            mask = (torchvision.io.read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY) > 127) * 1
            
            h, w = image.shape[-2:]

            # perform fill hole operation on the background
            if fill_hole:
                labeled = skimage.measure.label(label[0], background=1, connectivity=1 if eight_connectivity else 2)    # using background=1 and connectivity=1 to label the background when eight_connectivity is True for the foreground
                labeled = skimage.morphology.remove_small_objects(labeled, min_size=6, connectivity=1 if eight_connectivity else 2)
                label = torch.tensor(labeled == 0, dtype=torch.float32)
            else:
                label = label[0]

            mask = mask[0].unsqueeze(dim=0)
            label = torch.stack((1-label, label), dim=0).squeeze(dim=1)

            if not augment and not five_crop:
                # crop 4 corners from center from the image and append to list
                center_x, center_y = h // 2, w // 2
                image1 = TVF.crop(image, center_x - crop_size, center_y - crop_size, crop_size, crop_size)
                label1 = TVF.crop(label, center_x - crop_size, center_y - crop_size, crop_size, crop_size)
                mask1 = TVF.crop(mask, center_x - crop_size, center_y - crop_size, crop_size, crop_size)

                image2 = TVF.crop(image, center_x - crop_size, center_y, crop_size, crop_size)
                label2 = TVF.crop(label, center_x - crop_size, center_y, crop_size, crop_size)
                mask2 = TVF.crop(mask, center_x - crop_size, center_y, crop_size, crop_size)

                image3 = TVF.crop(image, center_x, center_y - crop_size, crop_size, crop_size)
                label3 = TVF.crop(label, center_x, center_y - crop_size, crop_size, crop_size)
                mask3 = TVF.crop(mask, center_x, center_y - crop_size, crop_size, crop_size)

                image4 = TVF.crop(image, center_x, center_y, crop_size, crop_size)
                label4 = TVF.crop(label, center_x, center_y, crop_size, crop_size)
                mask4 = TVF.crop(mask, center_x, center_y, crop_size, crop_size)
                
                self.images.extend([image1, image2, image3, image4])
                self.labels.extend([label1, label2, label3, label4])
                self.masks.extend([mask1, mask2, mask3, mask4])
            else:
                self.images.append(image)
                self.labels.append(label)
                self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        mask = self.masks[idx]
        im_h, im_w = image.shape[-2:]

        # RandomCrop
        if self.augment:
            crop_size = self.crop_size
            if self.five_crop:
                crop_size *= 2
            i, j, h, w = v2.RandomCrop.get_params(
                image, output_size=(min(crop_size, im_h), min(crop_size, im_w)))

            image = TVF.crop(image, i, j, h, w)
            label = TVF.crop(label, i, j, h, w)
            mask = TVF.crop(mask, i, j, h, w)
            
            #rotate
            angle = random.choice([0, 90, 180, 270])
            image = TVF.rotate(image, angle)
            label = TVF.rotate(label, angle)
            mask = TVF.rotate(mask, angle)
        elif not self.five_crop:
            image = TVF.center_crop(image, self.crop_size)
            label = TVF.center_crop(label, self.crop_size)
            mask = TVF.center_crop(mask, self.crop_size)

        image = self.transforms(image)
        label = self.target_transforms(label)
        mask = self.target_transforms(mask)
        if self.five_crop:
            image = torch.stack(image, dim=0)
            label = torch.stack(label, dim=0)
            mask = torch.stack(mask, dim=0)

        return {"img": image, "seg": label, "mask": mask[0]}
    



class DriveFullData(Dataset):
    def __init__(self, train_path, test_path):
        # set: "train" or "val"
        img_dir_train = os.path.join(train_path, "images")
        label_dir_train = os.path.join(train_path, "1st_manual")

        img_dir_test = os.path.join(test_path, "images")
        label_dir_test = os.path.join(test_path, "1st_manual")

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