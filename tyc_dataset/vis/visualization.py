from typing import Tuple

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import Tensor

from ..data.utils import normalize_0_1


def plot_image_instances(
        image: Tensor,
        instances: Tensor,
        class_labels: Tensor,
        save: bool = False,
        show: bool = False,
        file_path: str = "plot.png",
        alpha: float = 0.5,
        colors_cells: Tuple[Tuple[float, float, float], ...] = (
                (1.0, 0.0, 0.89019608),
                (1.0, 0.5, 0.90980392),
                (0.7, 0.0, 0.70980392),
                (0.7, 0.5, 0.73333333),
                (0.5, 0.0, 0.53333333),
                (0.5, 0.2, 0.55294118),
                (0.3, 0.0, 0.45),
                (0.3, 0.2, 0.45),
        ),
        colors_traps: Tuple[Tuple[float, float, float], ...] = ((0.05, 0.05, 0.05), (0.25, 0.25, 0.25)),
        cell_class: int = 1,
) -> None:
    """Plots the instance segmentation label overlaid with the microscopy image.

    Args:
        image (Tensor): Input image of shape [1, H, W].
        instances (Tensor): Instances masks of shape [N, W, W].
        class_labels (Tensor): Class labels of each instance [N].
        save (bool): If true image will be stored under given path name. Default False.
        show (bool): If true plt.show() will be called. Default False.
        file_path (str): Path and name where image will be stored. Default "plot.png".
        alpha (float): Transparency factor of the instances. Default 0.3.
        colors_cells (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each trap instances.
        colors_traps (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each cell instances.
        cell_class int: Tuple of cell classes. Default 1.
    """
    # Normalize image to [0, 1]
    image = normalize_0_1(image).repeat(3, 1, 1).permute(1, 2, 0).detach().cpu()
    # Convert data to numpy
    instances = instances.detach().cpu()
    class_labels = class_labels.detach().cpu()
    # Cell colors to torch tensor
    colors_cells = torch.tensor(colors_cells)
    colors_traps = torch.tensor(colors_traps)
    # Get semantic segmentation map
    semantic_segmentation: Tensor = (instances * class_labels.view(-1, 1, 1)).sum(dim=0)
    # Get index map
    instances = (instances * torch.arange(start=1, end=instances.shape[0] + 1)[:, None, None]).sum(dim=0)
    # Make RGB instance segmentation map
    max_id = instances.max().item()
    embedding_weights_cell = torch.cat([colors_cells for _ in range(math.ceil(max_id / colors_cells.shape[0]) + 1)],
                                       dim=0)
    embedding_weights_cell = torch.cat([torch.tensor([[1., 1., 1.]]), embedding_weights_cell], dim=0)
    rgb_instance_segmentation_cell = torch.embedding(embedding_weights_cell, instances.long())

    embedding_weights_trap = torch.cat([colors_traps for _ in range(math.ceil(max_id / colors_traps.shape[0]) + 1)],
                                       dim=0)
    embedding_weights_trap = torch.cat([torch.tensor([[1., 1., 1.]]), embedding_weights_trap], dim=0)
    rgb_instance_segmentation_trap = torch.embedding(embedding_weights_trap, instances.long())
    rgb_instance_segmentation = torch.where(semantic_segmentation[..., None] == cell_class,
                                            rgb_instance_segmentation_cell, rgb_instance_segmentation_trap)
    # Generate overlay
    mask = (rgb_instance_segmentation == 1.0).all(dim=-1, keepdim=True).repeat(1, 1, 3)
    image = torch.where(mask, image, (1. - alpha) * image + alpha * rgb_instance_segmentation)
    if save:
        torchvision.utils.save_image(image.permute(2, 0, 1).float(), nrow=1, padding=0, normalize=True,
                                     fp=file_path)
    if show:
        # Init figure
        fig, ax = plt.subplots()
        # Set size
        fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
        # Plot image and instances
        ax.imshow(image.numpy())
        # Axis off
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Show figure
        plt.show(bbox_inches='tight', pad_inches=0)
        # Close figure
        plt.close()

def plot_instances(
        instances: Tensor,
        class_labels: Tensor,
        save: bool = False,
        show: bool = False,
        file_path: str = "",
        colors_cells: Tuple[Tuple[float, float, float], ...] = (
                (1.0, 0.0, 0.89019608),
                (1.0, 0.5, 0.90980392),
                (0.7, 0.0, 0.70980392),
                (0.7, 0.5, 0.73333333),
                (0.5, 0.0, 0.53333333),
                (0.5, 0.2, 0.55294118),
                (0.3, 0.0, 0.45),
                (0.3, 0.2, 0.45),
        ),
        colors_traps: Tuple[Tuple[float, float, float], ...] = ((0.3, 0.3, 0.3), (0.5, 0.5, 0.5)),
        cell_class: int = 1
) -> None:
    """Just plots the instance segmentation map.

    Args:
        instances (Tensor): Instances masks of shape [N, W, W].
        class_labels (Tensor): Class labels of each instance [N].
        save (bool): If true image will be stored under given path name. Default False.
        show (bool): If true plt.show() will be called. Default False.
        file_path (str): Path and name where image will be stored. Default "plot.png".
        colors_cells (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each trap instances.
        colors_traps (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each cell instances.
        cell_class int: Tuple of cell classes. Default 1.
    """
    # Convert data to numpy
    instances = instances.detach().cpu()
    class_labels = class_labels.detach().cpu()
    # Cell colors to torch tensor
    colors_cells = torch.tensor(colors_cells)
    colors_traps = torch.tensor(colors_traps)
    # Get semantic segmentation map
    semantic_segmentation: Tensor = (instances * class_labels.view(-1, 1, 1)).sum(dim=0)
    # Get index map
    instances = (instances * torch.arange(start=1, end=instances.shape[0] + 1)[:, None, None]).sum(dim=0)
    # Make RGB instance segmentation map
    max_id = instances.max().item()
    embedding_weights_cell = torch.cat([colors_cells for _ in range(math.ceil(max_id / colors_cells.shape[0]) + 1)],
                                       dim=0)
    embedding_weights_cell = torch.cat([torch.tensor([[1., 1., 1.]]), embedding_weights_cell], dim=0)
    rgb_instance_segmentation_cell = torch.embedding(embedding_weights_cell, instances.long())

    embedding_weights_trap = torch.cat([colors_traps for _ in range(math.ceil(max_id / colors_traps.shape[0]) + 1)],
                                       dim=0)
    embedding_weights_trap = torch.cat([torch.tensor([[1., 1., 1.]]), embedding_weights_trap], dim=0)
    rgb_instance_segmentation_trap = torch.embedding(embedding_weights_trap, instances.long())
    rgb_instance_segmentation = torch.where(semantic_segmentation[..., None] == cell_class,
                                            rgb_instance_segmentation_cell, rgb_instance_segmentation_trap)
    # Permute to [3, H, W]
    rgb_instance_segmentation = rgb_instance_segmentation.permute(2, 0, 1)
    if save:
        torchvision.utils.save_image(rgb_instance_segmentation.float(), nrow=1, padding=0, normalize=True,
                                     fp=file_path)
    if show:
        # Init figure
        fig, ax = plt.subplots()
        # Set size
        fig.set_size_inches(5, 5 * rgb_instance_segmentation.shape[0] / rgb_instance_segmentation.shape[1])
        # Plot image and instances
        ax.imshow(rgb_instance_segmentation.permute(1, 2, 0).numpy())
        # Axis off
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Show figure
        plt.show(bbox_inches='tight', pad_inches=0)
        # Close figure
        plt.close()