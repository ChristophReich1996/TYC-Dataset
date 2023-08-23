from typing import List, Optional, Tuple

import os
import json

import cv2
import kornia.augmentation
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class TYCDataset(Dataset):
    """This class implements the TYC dataset."""

    def __init__(
            self,
            path: str,
            augmentations: Optional[kornia.augmentation.AugmentationSequential] = None,
    ) -> None:
        """Constructor method.

        Args:
            path (str): Path to dataset.
            augmentations (Optional[kornia.augmentation.AugmentationSequential]): Augmentations. Default None.
        """
        # Call super constructor
        super(TYCDataset, self).__init__()
        # Save parameters
        self.transforms: Optional[kornia.augmentation.AugmentationSequential] = augmentations
        # Check augmentations
        self._check_transforms()
        # Get paths of input images
        self.inputs: List[str] = self._get_files(os.path.join(path, "images"))
        # Get paths of instance masks
        self.classes: List[str] = self._get_files(os.path.join(path, "labels", "classes"))
        # Get paths of class labels
        self.masks: List[str] = self._get_files(os.path.join(path, "labels", "masks"))

    def _check_transforms(self) -> None:
        """Checks if transformations are valid.

        Raises:
            RuntimeError if transformations are not correctly configured.
        """
        # If no transformation is given we have a valid case
        if self.transforms is None:
            return
        # Check if augmentations include all keys
        if (
                (self.transforms.data_keys[0].value == 0)
                and (self.transforms.data_keys[1].value == 1)
        ):
            return
        raise RuntimeError("Transforms must include the data keys: [''input'', ''mask''].")

    def _get_files(self, path: str) -> List[str]:
        """Gets all files in a given path.

        Args:
            path (str): Path to search in.

        Returns:
            files (List[str]): List of all files in path.
        """
        files: List[str] = []
        for file in sorted(os.listdir(path)):
            if (not file.startswith(".")) and (os.path.isfile(os.path.join(path, file))):
                files.append(os.path.join(path, file))
        return files

    def __len__(self) -> int:
        """Method returns the length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        return len(self.inputs)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Method returns an instance of the dataset.

        Notes:
            class_labels is a one-hot vector.
            The semantic class of traps is 0 the semantic class of cells is 1.

        Args:
            item (int): Index of the dataset instance

        Returns:
            image (Tensor): Image if the shape [1, H, W].
            instances (Tensor): Instance maps of the shape [N, H, W].
            class_labels (Tensor): Class labels of the shape [N, C].
        """
        # Load input images
        image = torch.from_numpy(cv2.imread(self.inputs[item], -1).astype(float)).unsqueeze(dim=0).float()
        # Load instance masks (indexes)
        instances = torch.from_numpy(cv2.imread(self.masks[item], -1).astype(int)).long()
        instances = instances[..., 2] + 256 * instances[..., 1]
        # Instance masks to binary maps
        instances = one_hot(instances, int(instances.max() + 1)).float().permute(2, 0, 1)
        # Eliminate all zero maps and background mask
        mask = instances.sum(dim=(1, 2)) > 0.0
        instances = instances[mask]
        instances = instances[1:]
        with open(self.classes[item]) as file:
            classes = json.load(file)["annotations"]
        # Concert list of dicts with {"id":0, "category_id":1} to {id: category_ids}
        classes = one_hot(torch.tensor([entry["category_id"] for entry in classes]), 2)
        # Sanity check
        assert instances.shape[0] == classes.shape[0], "Missmatch between instance mask and class instance label!"
        # Apply transformations
        if self.transforms:
            tensors = self.transforms(image[None], instances[None])
            image, instances = tensors[0][0], tensors[1][0]
        return image, instances, classes


def collate_function_tyc_dataset(
        batch: List[Tuple[Tensor]],
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Custom collate function for YIM dataset.

    Args:
        batch (Tuple[Iterable[Tensor], Iterable[Tensor], Iterable[Tensor], Iterable[Tensor]]):
            Batch of input data, instances maps, bounding boxes and class labels

    Returns:
        images (List[Tensor]): List images of the shape [B, 1, H, W]. Image shape can differ between images.
        instances (List[Tensor]): List of instance maps as tensors with shape [B, H, W] each.
        class_labels (List[Tensor]): Class labels as a list of tensors with shape [N, C].
    """
    return (
        [input_samples[0] for input_samples in batch],
        [input_samples[1] for input_samples in batch],
        [input_samples[2] for input_samples in batch],
    )
