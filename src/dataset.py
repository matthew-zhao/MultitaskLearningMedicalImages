import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, transform=None, second_transform=None, albumentations_transforms=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform
        self.albumentations_transforms = albumentations_transforms
        self.second_transform = second_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        images = []
        labels = []
        for i in range(count):
            image = pil_loader(study_path + 'image%s.png' % (i+1))
            padded_image = self.pad_image(image)
            augmented_image = self.transform(padded_image)
            if self.albumentations_transforms:
                augmented_image = self.albumentations_transforms(image=np.array(augmented_image))['image']
            else:
                augmented_image = np.array(augmented_image)
            # subtract mean of image and divide by (max - min) range
            preprocessed_image = self.preprocess_input(augmented_image)
            #preprocessed_image = self.channels_last_to_first(preprocessed_image)
            images.append(self.second_transform(preprocessed_image))
            labels.append(label)
        images = torch.stack(images)
        labels = torch.from_numpy(np.array(labels)).long()
        return images, labels

    #def channels_last_to_first(self, img):
    #    """ Move the channels to the first dimension."""
    #    img = np.swapaxes(img, 0,2)
    #    img = np.swapaxes(img, 1,2)
    #    return img

    def preprocess_input(self, img):
        """ Preprocess an input image. """
        # assume image is RGB
        img = img[..., ::-1].astype('float32')
        img_min = float(np.min(img)) ; img_max = float(np.max(img))
        img_range = img_max - img_min
        if img_range == 0: img_range = 1.
        img = (img - img_min) / img_range
        #img[..., 0] -= 0.485
        #img[..., 1] -= 0.456
        #img[..., 2] -= 0.406
        #img[..., 0] /= 0.229
        #img[..., 1] /= 0.224
        #img[..., 2] /= 0.225
        return img

    def pad_image(self, img, ratio=1.):
        # Default is ratio=1 aka pad to create square image
        ratio = float(ratio)
        # Given ratio, what should the height be given the width?
        img = np.array(img)
        h, w = img.shape[:2]
        #print(h, w)
        desired_h = int(w * ratio)
        # If the height should be greater than it is, then pad top/bottom
        if desired_h > h:
            hdiff = int(desired_h - h)
            pad_list = [(hdiff // 2, desired_h-h-(hdiff // 2)), (0,0), (0,0)]
        # If height is smaller than it is, then pad left/right
        elif desired_h < h:
            desired_w = int(h / ratio)
            wdiff = int(desired_w - w)
            pad_list = [(0,0), (wdiff // 2, desired_w-w-(wdiff // 2)), (0,0)]
        elif desired_h == h:
            return img
        #print(pad_list)
        return np.pad(img, pad_list, 'constant', constant_values=np.min(img))