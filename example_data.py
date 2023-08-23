import kornia.augmentation
from torch.utils.data import DataLoader

import tyc_dataset


DATASET_PATH: str = "/Users/christoph/Desktop/tyc/labeled_set/train"

def main() -> None:
    # Init augmentations
    transforms = kornia.augmentation.AugmentationSequential(
        kornia.augmentation.RandomHorizontalFlip(p=1.0),
        data_keys=["input", "mask"],
        same_on_batch=False,
    )
    # Init dataset
    dataset = tyc_dataset.data.TYCDataset(path=DATASET_PATH, augmentations=transforms)
    for index, (images, instances, class_labels) in enumerate(dataset):
        print(index, images.shape, instances.shape, class_labels.shape)
    # Make data loader
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=2,
        batch_size=2,
        drop_last=False,
        collate_fn=tyc_dataset.data.collate_function_tyc_dataset,
    )
    # Loop over data loader
    for index, (images, instances, class_labels) in enumerate(data_loader):
        print(index, len(images), len(instances), len(class_labels))


if __name__ == "__main__":
    main()
