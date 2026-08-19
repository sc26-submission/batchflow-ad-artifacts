from __future__ import annotations

import re
import time
from typing import Callable

import torch
from PIL import Image
from torchvision import transforms


class DelayTransform:
    def __init__(self, delay_ms: float) -> None:
        self.delay_seconds = max(0.0, delay_ms / 1000.0)

    def __call__(self, x):
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        return x


class ALBEFTextTransform:
    """ALBEF BERT tokenization used by the COCO retrieval workload."""

    def __init__(
        self,
        *,
        pretrained_tokenizer: str = "bert-base-uncased",
        max_seq_len: int = 30,
        add_end_token: bool = False,
        pad_to_max_seq_len: bool = True,
    ) -> None:
        try:
            from transformers import BertTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ALBEF text preprocessing requires the 'transformers' package"
            ) from exc

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer)
        self.max_seq_len = max_seq_len
        self.add_end_token = add_end_token
        self.pad_to_max_seq_len = pad_to_max_seq_len

    @staticmethod
    def _preprocess(text: str) -> str:
        return (
            re.sub(r"([,.'!?\"()*#:;~])", "", text)
            .replace("-", " ")
            .replace("/", " ")
            .rstrip(" ")
        )

    def __call__(self, text: str) -> torch.Tensor:
        # Preserve the preprocessing used by the older W3 implementation.
        # Its add_end_token=False setting maps directly to HuggingFace's
        # add_special_tokens=False, followed by truncation and fixed padding.
        encoded = self.tokenizer(
            self._preprocess(text),
            max_length=self.max_seq_len,
            padding="max_length" if self.pad_to_max_seq_len else False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=self.add_end_token,
        )
        return encoded["input_ids"].squeeze(0).long()


def build_text_transform(name: str) -> Callable[[str], torch.Tensor]:
    if name == "albef_text_30":
        return ALBEFTextTransform(
            max_seq_len=30,
            add_end_token=False,
            pad_to_max_seq_len=True,
        )
    raise ValueError(f"unknown text transform name: {name}")


def build_image_transform(
    name: str,
    *,
    delay_ms: float = 0.0,
) -> Callable[[Image.Image], torch.Tensor]:
    delay = DelayTransform(delay_ms)

    if name == "resize_224":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            delay,
        ])

    if name == "resize_224_uint8":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            delay,
        ])

    if name == "center_crop_224":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            delay,
        ])

    if name == "center_crop_224_uint8":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            delay,
        ])

    if name == "imagenet_train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            delay,
        ])

    if name == "imagenet_train_uint8":
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            delay,
        ])

    if name == "imagenet_eval":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            delay,
        ])

    if name == "openimages_train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            delay,
        ])

    if name == "openimages_eval":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            delay,
        ])

    if name == "albef_train_384":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                384,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 7),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
            delay,
        ])

    if name == "albef_eval_384":
        return transforms.Compose([
            transforms.Resize(
                (384, 384),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
            delay,
        ])

    if name == "cifar10_train":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            delay,
        ])

    if name == "cifar10_eval":
        return transforms.Compose([
            transforms.ToTensor(),
            delay,
        ])

    if name == "cifar10_train_uint8":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            delay,
        ])

    if name == "light_aug":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            delay,
        ])

    if name == "light_aug_uint8":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            delay,
        ])

    if name == "heavy_aug":
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            transforms.ToTensor(),
            delay,
        ])

    if name == "blur_aug":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),
            delay,
        ])

    if name == "none":
        return transforms.Compose([
            transforms.ToTensor(),
            delay,
        ])

    if name == "identity_uint8":
        return transforms.Compose([
            transforms.PILToTensor(),
            delay,
        ])

    raise ValueError(f"unknown transform name: {name}")
