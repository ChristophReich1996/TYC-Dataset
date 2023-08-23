import kornia.augmentation
from torch.utils.data import DataLoader

import tyc_dataset

DATASET_PATH: str = "/Users/christoph/Desktop/tyc/labeled_set/val"


def main() -> None:
    # Init augmentations
    transforms = kornia.augmentation.AugmentationSequential(
        kornia.augmentation.RandomHorizontalFlip(p=1.0),
        data_keys=["input", "mask"],
        same_on_batch=False,
    )
    # Init dataset
    dataset = tyc_dataset.data.TYCDataset(path=DATASET_PATH, augmentations=transforms)
    # Make data loader
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=8,
        batch_size=1,  # Only works with a batch size of 1 due to the different image shapes
        drop_last=True,
        collate_fn=tyc_dataset.data.collate_function_tyc_dataset,
    )
    # Init PQ
    pq = tyc_dataset.eval.PanopticQuality()
    cell_iou = tyc_dataset.eval.CellIoU()
    # Loop over data loader
    for index, (images, instances, class_labels) in enumerate(data_loader):
        # Get semantic classes form one-hot vector
        semantic_classes = [c.argmax(dim=-1) for c in class_labels]
        # Shift the label just a little to simulate a near perfect prediction
        instances_pred = [i.clone().roll(shifts=(5, 5), dims=(1, 2)) for i in instances]
        # Copy list
        semantic_classes_pred = semantic_classes.copy()
        # Simulate case where no instance was detected
        if index == 0:
            instances_pred[0] = None
            semantic_classes_pred[0] = None
        # Compute metrics
        pq.update(
            instances_pred=instances_pred,
            classes_pred=semantic_classes_pred,
            instances_target=instances,
            classes_target=semantic_classes,
        )
        cell_iou.update(
            instances_pred=instances_pred,
            classes_pred=semantic_classes_pred,
            instances_target=instances,
            classes_target=semantic_classes,
        )
    print(f"Panoptic Quality: {pq.compute().item()}")
    print(f"Cell class IoU: {cell_iou.compute().item()}")


if __name__ == "__main__":
    main()
