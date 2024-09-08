import torch
import cv2
import numpy as np

from config import CLASSES, RESIZE_TO, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform, generate_image_dataset


class GeneratedImagesDataset(Dataset):
    def __init__(self, width, height, classes, transforms=None):
        self.transforms = transforms
        self.height = height
        self.width = width
        self.classes = classes

    def __getitem__(self, idx):

        # Generate the image and annotations on the fly
        data = generate_image_dataset('../masa.png', '../poze_jucarii', '../poze_nonjucarii', 1)

        if data is None:
            raise FileNotFoundError(f"The image file could not be generated.")

        img, boxes = data[0][0], data[1][0]
        img = np.array(img)
        img_resized = cv2.resize(img, (self.width, self.height))
        img_resized = img_resized.astype(np.float32) / 255.0

        # Normalize the bounding box coordinates here
        labels = [box[4] for box in boxes]
        boxes = torch.tensor([box[:4] for box in boxes]).float()

        # Get the height and width of the image
        image_width = img.shape[1]
        image_height = img.shape[0]

        boxes_resize = []

        for box in boxes:
            xmin, ymin, xmax, ymax = box

            # Resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height

            boxes_resize.append([xmin_final, ymin_final, xmax_final, ymax_final])


        # Bounding box to tensor
        boxes_resize = torch.tensor(boxes_resize, dtype=torch.float32)

        # Area of the bounding boxes
        area = (boxes_resize[:, 3] - boxes_resize[:, 1]) * (boxes_resize[:, 2] - boxes_resize[:, 0])

        # No crowd instances
        iscrowd = torch.zeros((boxes_resize.shape[0],), dtype=torch.int64)

        # Labels to tensor
        labels_tensor = torch.ones((boxes_resize.shape[0],), dtype=torch.int64)
        for index in range(len(labels)):
            if labels[index] == 'Non-Jucarie':
                labels_tensor[index] += 1

        # Prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes_resize
        target["labels"] = labels_tensor
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Apply the image transforms
        if self.transforms:
            sample = self.transforms(image=img_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            img_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_resized, target

    def __len__(self):
        return 500  # Number of images to generate


# Prepare the final datasets and data loaders
train_dataset = GeneratedImagesDataset(RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = GeneratedImagesDataset(RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

if __name__ == '__main__':

    # Sanity check of the Dataset pipeline with sample visualization
    dataset = GeneratedImagesDataset(RESIZE_TO, RESIZE_TO, CLASSES)
    print(f"Number of training images: {len(dataset)}")


    # Function to visualize a single sample
    def visualize_sample(image, target):
        boxes = target['boxes']
        labels = target['labels']
        # label = CLASSES[label.item()] ## FOR ONLY ONE CLASS!!

        for box, label in zip(boxes, labels):
            label = CLASSES[label.item()]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Image', image)
        cv2.waitKey(0)


    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)
