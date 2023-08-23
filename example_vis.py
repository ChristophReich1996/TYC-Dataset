import kornia.augmentation

import tyc_dataset

DATASET_PATH: str = "/Users/christoph/Desktop/tyc/labeled_set/test_ood"


def main() -> None:
    # Init augmentations
    transforms = kornia.augmentation.AugmentationSequential(
        kornia.augmentation.RandomHorizontalFlip(p=1.0),
        kornia.augmentation.RandomVerticalFlip(p=1.0),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(31, 31), sigma=(9, 9), p=1.0),
        data_keys=["input", "mask"],
        same_on_batch=False,
    )
    # Init dataset
    dataset = tyc_dataset.data.TYCDataset(path=DATASET_PATH, augmentations=transforms)
    # Get sample from dataset
    image, instances, class_labels = dataset[6]
    # Plotting (we need to add one to the class label to consider the background)
    tyc_dataset.vis.plot_image_instances(
        image=image, instances=instances, class_labels=class_labels.argmax(dim=1) + 1, save=True, show=True,
        file_path="plot_instance_seg_overlay.png"
    )
    tyc_dataset.vis.plot_instances(instances=instances, class_labels=class_labels.argmax(dim=1) + 1, save=True,
                                   show=True, file_path="plot_instance_seg.png")


if __name__ == "__main__":
    main()
