import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
from transformers import BertTokenizer, BertModel
import torch

# 初始化BERT tokenizer和model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')



def get_bert_embedding(text):
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get embeddings
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Use last hidden state as embeddings
    embeddings = outputs.last_hidden_state.squeeze(0)
    return embeddings.numpy()


def map_labels_to_4class(mask):
    """
    将标签映射为4个类别：
    0 → 0
    63 → 1
    127 → 2
    255 → 3
    其他值按最近邻映射
    """
    # 创建映射表
    label_map = np.zeros(256, dtype=np.uint8)
    label_map[0] = 0  # 背景
    label_map[63] = 1  # 类别1
    label_map[127] = 2  # 类别2
    label_map[255] = 3  # 类别3

    # 对于不在指定值中的像素，找到最接近的映射值
    for i in range(1, 256):
        if i not in [0, 63, 127, 255]:
            distances = [abs(i - 0), abs(i - 63), abs(i - 127), abs(i - 255)]
            closest_class = np.argmin(distances)
            label_map[i] = closest_class

    # 应用映射
    return label_map[mask]


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):
        mask_filename = self.mask_list[idx]
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # 应用4类别映射
        mask = map_labels_to_4class(mask)

        mask = correct_dims(mask)
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        text_embedding = get_bert_embedding(text)
        if text_embedding.shape[0] > 14:
            text_embedding = text_embedding[:14, :]

        if self.one_hot_mask:
            assert self.one_hot_mask == 4, 'one_hot_mask must be 4 for 4-class classification'
            mask = torch.zeros((4, mask.shape[1], mask.shape[2])).scatter_(0, torch.from_numpy(mask).long(), 1)

        sample = {'label': mask, 'text': text_embedding}

        return sample, mask_filename


class ImageToImage2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        mask_filename = image_filename[: -3] + "png"
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # 应用4类别映射
        mask = map_labels_to_4class(mask)

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        text_embedding = get_bert_embedding(text)
        if text_embedding.shape[0] > 10:
            text_embedding = text_embedding[:10, :]

        if self.one_hot_mask:
            assert self.one_hot_mask == 4, 'one_hot_mask must be 4 for 4-class classification'
            mask_tensor = torch.from_numpy(mask).long()
            mask = torch.zeros((4, mask.shape[1], mask.shape[2])).scatter_(0, mask_tensor, 1)

        sample = {'image': image, 'label': mask, 'text': text_embedding}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename