# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .citypersons import CityPersonsDataset
from .cityscapes import CityScapesDataset
from .coco import COCODataset
from .crowdhuman import CrowdHumanDataset
from .imagenet import ImageNetDataset
from .lvis import LVISDataset
from .objects365 import Objects365Dataset
from .voc import VOCDataset
from .widerface import WiderFaceDataset
from .vg import  VGStanfordDataset
from .open_image import OpenImageDataset
from .gqa import GQADataset

__all__ = [
    "COCODataset",
    "VOCDataset",
    "CityScapesDataset",
    "ImageNetDataset",
    "WiderFaceDataset",
    "LVISDataset",
    "CityPersonsDataset",
    "Objects365Dataset",
    "VGStanfordDataset",
    "OpenImageDataset",
    "GQADataset"
]
