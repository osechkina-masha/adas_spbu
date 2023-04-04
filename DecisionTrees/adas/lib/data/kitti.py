import os
import typing as tp
from pathlib import Path

import cv2 as cv
from numpy import ndarray
from torch.utils.data import Dataset
from tqdm import trange

from .utils import get_filenames, segmentation_to_edge


class KittiEdges(Dataset):
    def __init__(self, path: str, holdout: str, pic_size: tp.Tuple[int, int] = (375, 1242)) -> None:
        if holdout == 'train':
            holdout_path = "training"
        elif holdout == "test":
            holdout_path = "testing"
        else:
            raise KeyError("Unknown holdout")

        self._pic_size = pic_size
        
        self._pic_folder_path = os.path.join(path, holdout_path, "image_2")
        self._segmentation_folder_path = os.path.join(path, holdout_path, "gt_image_2")
        self._contour_folder_path = os.path.join(path, holdout_path, f"edge_images_pic_size_{pic_size[0]}_{pic_size[1]}")

        self._image_names = sorted(get_filenames(self._pic_folder_path))
        self._segmentation_names = sorted(get_filenames(self._segmentation_folder_path))
        self._n_images = len(self._image_names)
        
        if not os.path.isdir(self._contour_folder_path):
            print("Contours not found. Building...")
            self._setup_contours()

    def _setup_contours(self):
        Path(self._contour_folder_path).mkdir(parents=True, exist_ok=True)
        for img_ind in trange(0, self._n_images):
            segmentation_path = self._segmentation_path(img_ind)
            segmentation = cv.imread(segmentation_path, cv.IMREAD_GRAYSCALE)
            segmentation = cv.resize(segmentation, self._pic_size[::-1])
            edges = segmentation_to_edge(segmentation)
            edge_path = self._contour_path(img_ind)
            cv.imwrite(edge_path, edges)

    def __getitem__(self, index) -> tp.Tuple[ndarray, ndarray]:
        img = cv.imread(self._img_path(index), cv.IMREAD_COLOR)
        img = cv.resize(img, self._pic_size[::-1])
        contours = cv.imread(self._contour_path(index), cv.IMREAD_GRAYSCALE)
        return img, contours
    
    def _img_path(self, i) -> str:
        return os.path.join(self._pic_folder_path, self._image_names[i])

    def _segmentation_path(self, i) -> str:
        return os.path.join(self._segmentation_folder_path, self._segmentation_names[i])

    def _contour_path(self, i) -> str:
        return os.path.join(self._contour_folder_path, f"{i}_edge.png")

    def __len__(self) -> int:
        return self._n_images
