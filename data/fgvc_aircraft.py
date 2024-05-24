import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .text import TEXT, CLASS_NAME


class FGVC_Aircraft(Dataset):

    def __init__(self, root, split='train',
                 transform=None, target_transform=None):

        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self._image_files = []
        self._labels = []
        self._class_names = []

        self.images_path = {}   # 11788
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path
        
        self.full_class = {}
        with open(os.path.join(self.root, 'classes.txt')) as f:
            for line in f:
                class_id = line.strip().split(' ')[0]
                class_name = ' '.join(line.strip().split(' ')[1:])
                self.full_class[class_id] = class_name
    

        img_label = {}
        with open(self.root + "/image_class_labels.txt", "r") as f:
            for line in f:
                img_id, label = line.strip().split(' ')
                img_label[img_id] = label

        if self.split == "train":
            with open(self.root + "/trainval_img_id.txt", "r") as f:
                for line in f:
                    img_id = line.strip()
                    img_path = os.path.join(self.root, 'images', self.images_path[img_id])
                    class_name = self.full_class[img_label[img_id]]
                    self._image_files.append(img_path)
                    self._labels.append(img_label[img_id])
                    self._class_names.append(class_name)

        elif self.split == "test_seen":
            with open(self.root + "/test_seen_img_id.txt", "r") as f:
                for line in f:
                    img_id = line.strip()
                    img_path = os.path.join(self.root, 'images', self.images_path[img_id])
                    class_name = self.full_class[img_label[img_id]]
                    self._image_files.append(img_path)
                    self._labels.append(img_label[img_id])
                    self._class_names.append(class_name)
        else:
            with open(self.root + "/test_unseen_img_id.txt", "r") as f:
                for line in f:
                    img_id = line.strip()
                    img_path = os.path.join(self.root, 'images', self.images_path[img_id])
                    class_name = self.full_class[img_label[img_id]]
                    self._image_files.append(img_path)
                    self._labels.append(img_label[img_id])
                    self._class_names.append(class_name)

        self.labels = np.array([int(i) for i in self._labels]) - 1
        self.classes = list(self.full_class.values())

        self.CLS2TXT = {key: value for key, value in zip(CLASS_NAME['fgvc-aircraft'], TEXT['fgvc-aircraft'])}
        self.class_texts = [self.CLS2TXT[cls] for cls in self.classes]
        
        with open(self.root + "/trainval_img_id.txt", "r") as f:
            seen_labels = [img_label[line.strip()] for line in f] 
        self.seen_idx = np.array([1 if str(i+1) in seen_labels else 0 for i in range(100)])

    def __len__(self):
        return len(self._image_files)
    
    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_path = self._image_files[index]
        class_id = self._labels[index]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)

        class_name = self.full_class[class_id]

        return image, int(class_id)-1, self.CLS2TXT[class_name]

    


if __name__ == '__main__':

    root = '../datasets/fgvc-aircraft-2013b'
    train_dataset = FGVC_Aircraft(root, split='train')
    print(len(train_dataset))
    test_seen_dataset = FGVC_Aircraft(root, split='test_seen')
    print(len(test_seen_dataset))
    test_unseen_dataset = FGVC_Aircraft(root, split='test_unseen')
    print(len(test_unseen_dataset))