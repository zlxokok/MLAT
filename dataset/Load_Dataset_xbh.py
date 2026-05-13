import numpy as np
import torch
import random
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from typing import Callable
import os
import cv2
from scipy import ndimage
import torch

# 使用本地BERT模型路径
from transformers import BertTokenizer, BertModel

# 使用本地路径加载BERT模型
bert_path = '/media/share/data5/2511110091/MTGT/dataset/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_model = BertModel.from_pretrained(bert_path)

def get_bert_embedding(text):
    """获取BERT embedding"""
    # Tokenize text
    if isinstance(text, list):
        text = ' '.join(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get embeddings
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Use last hidden state as embeddings
    embeddings = outputs.last_hidden_state.squeeze(0)
    return embeddings.numpy().astype(np.float32)


def random_rot_flip(image, label):
    """随机旋转和翻转"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """随机旋转"""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        text = sample['text']

        # 确保数据类型正确
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if label.dtype != np.int64:
            label = label.astype(np.int64)

        # 数据增强
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 获取当前尺寸
        h, w = image.shape[:2]
        target_h, target_w = self.output_size

        # 缩放
        if h != target_h or w != target_w:
            scale_h = target_h / h
            scale_w = target_w / w

            # 图像使用三次插值
            if len(image.shape) == 3:
                image = zoom(image, (scale_h, scale_w, 1), order=3)
            else:
                image = zoom(image, (scale_h, scale_w), order=3)

            # 标签使用最近邻插值（保持实例ID）
            if len(label.shape) == 3:
                label = zoom(label, (scale_h, scale_w, 1), order=0)
            else:
                label = zoom(label, (scale_h, scale_w), order=0)

        # 转换为tensor
        if len(image.shape) == 2:
            image = torch.from_numpy(image).unsqueeze(0).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        label = torch.from_numpy(label).long()
        text = torch.from_numpy(text).float()

        # 确保label的shape是 (H, W) 而不是 (H, W, 1)
        if len(label.shape) == 3 and label.shape[2] == 1:
            label = label.squeeze(-1)

        sample = {'image': image, 'label': label, 'text': text}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        text = sample['text']

        # 确保数据类型正确
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if label.dtype != np.int64:
            label = label.astype(np.int64)

        # 获取当前尺寸
        h, w = image.shape[:2]
        target_h, target_w = self.output_size

        # 缩放
        if h != target_h or w != target_w:
            scale_h = target_h / h
            scale_w = target_w / w

            # 图像使用三次插值
            if len(image.shape) == 3:
                image = zoom(image, (scale_h, scale_w, 1), order=3)
            else:
                image = zoom(image, (scale_h, scale_w), order=3)

            # 标签使用最近邻插值
            if len(label.shape) == 3:
                label = zoom(label, (scale_h, scale_w, 1), order=0)
            else:
                label = zoom(label, (scale_h, scale_w), order=0)

        # 转换为tensor
        if len(image.shape) == 2:
            image = torch.from_numpy(image).unsqueeze(0).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        label = torch.from_numpy(label).long()
        text = torch.from_numpy(text).float()

        # 确保label的shape是 (H, W)
        if len(label.shape) == 3 and label.shape[2] == 1:
            label = label.squeeze(-1)

        sample = {'image': image, 'label': label, 'text': text}
        return sample


def correct_dims(*images):
    """确保图像维度正确"""
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


class ImageToImage2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: dict, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = sorted(os.listdir(self.input_path))
        self.mask_list = sorted(os.listdir(self.output_path))
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        # 根据数据集调整mask文件名
        if self.task_name == "MoNuSeg":
            mask_filename = image_filename[:-3] + "png"
        else:
            mask_filename = image_filename

        # 读取图像
        image_path = os.path.join(self.input_path, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0  # 归一化到 [0,1]

        # 读取mask - 保持原始实例ID
        mask_path = os.path.join(self.output_path, mask_filename)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                raise ValueError(f"无法读取mask: {mask_path}")
            # 使用最近邻插值保持实例ID
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.int64)
        else:
            print(f"警告: mask文件不存在 {mask_path}")
            mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)

        # 修正维度
        image, mask = correct_dims(image, mask)

        # 处理文本
        text = self.rowtext.get(mask_filename, "")
        if isinstance(text, list):
            text = text[0] if text else ""
        if not text:
            text = ""

        # 获取BERT embedding
        text_embedding = get_bert_embedding(text)
        if text_embedding.shape[0] > 10:
            text_embedding = text_embedding[:10, :]

        # one-hot编码（不推荐用于实例分割）
        if self.one_hot_mask:
            num_classes = int(mask.max()) + 1
            mask_tensor = torch.from_numpy(mask.squeeze()).long()
            mask = torch.zeros((num_classes, mask.shape[1], mask.shape[2])).scatter_(0, mask_tensor.unsqueeze(0), 1)

        sample = {'image': image, 'label': mask, 'text': text_embedding}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: dict, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = sorted(os.listdir(self.output_path))
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        mask_filename = self.mask_list[idx]

        # 读取mask - 保持原始实例ID
        mask_path = os.path.join(self.output_path, mask_filename)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise ValueError(f"无法读取mask: {mask_path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int64)
        mask = correct_dims(mask)

        # 处理文本
        text = self.rowtext.get(mask_filename, "")
        if isinstance(text, list):
            text = text[0] if text else ""
        if not text:
            text = ""

        text_embedding = get_bert_embedding(text)
        if text_embedding.shape[0] > 14:
            text_embedding = text_embedding[:14, :]

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            num_classes = int(mask.max()) + 1
            mask_tensor = torch.from_numpy(mask.squeeze()).long()
            mask = torch.zeros((num_classes, mask.shape[1], mask.shape[2])).scatter_(0, mask_tensor.unsqueeze(0), 1)

        sample = {'label': mask, 'text': text_embedding}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, mask_filename


# 测试代码
    # 测试BERT加

    # 创建测试数据集（需要替换为实际路径）
    # dataset = ImageToImage2D(
    #     dataset_path="/path/to/your/dataset",
    #     task_name="MoNuSeg",
    #     row_text={},
    #     joint_transform=RandomGenerator(output_size=(256, 256)),
    #     image_size=256
    # )
    #
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    #
    # for batch, names in dataloader:
    #     print(f"Image shape: {batch['image'].shape}")
    #     print(f"Label shape: {batch['label'].shape}")
    #     print(f"Label unique values: {torch.unique(batch['label'])}")
    #     break