import albumentations as A
import cv2
import numpy as np
import random
import os
from PIL import Image

from albumentations.pytorch import ToTensorV2
from config import DEVICE


# This class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# Define the training tranforms
def get_train_transform():
    return A.Compose([
        A.Flip(False),
        A.RandomRotate90(False),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


# Define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def show_tranformed_image(train_loader):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box in boxes:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Index to keep track of the number of images generated
index = 0


def generate_image(background_path, images_to_place_file_path, output_dir, annotations_dir, images_to_generate=1):
    for i in range(images_to_generate):

        number_of_images_to_place = random.randint(3, 7)

        img = Image.open(background_path)

        top_left_coordinates = (435, 59)
        bottom_right_coordinates = (1619, 934)

        images = [os.path.join(images_to_place_file_path, f) for f in os.listdir(images_to_place_file_path) if
                  f.endswith('.png')]

        selected_images = random.sample(images, number_of_images_to_place)

        occupied_coordinates = set()
        annotations_list = []

        for image_path in selected_images:
            image = Image.open(image_path)

            while True:

                position_x = random.uniform(0,
                                            1 - image.width / (bottom_right_coordinates[0] - top_left_coordinates[0]))
                position_y = random.uniform(0,
                                            1 - image.height / (bottom_right_coordinates[1] - top_left_coordinates[1]))

                position_x = int(
                    position_x * (bottom_right_coordinates[0] - top_left_coordinates[0]) + top_left_coordinates[0])
                position_y = int(
                    position_y * (bottom_right_coordinates[1] - top_left_coordinates[1]) + top_left_coordinates[1])

                # Check if the coordinates overlap with the ones already occupied
                new_area = set(
                    (x, y) for x in range(position_x, position_x + image.width) for y in
                    range(position_y, position_y + image.height)
                )

                if not new_area & occupied_coordinates:
                    occupied_coordinates.update(new_area)
                    break

            img.paste(image, (position_x, position_y))
            annotations_list.append(
                [position_x / img.width, position_y / img.height,
                 (position_x + image.width) / img.width, (position_y + image.height) / img.height]
            )

        # Generate a unique file name for each image
        output_path = os.path.join(output_dir, f"poza{i}.png")
        img.save(output_path)

        annotation_path = os.path.join(annotations_dir, f"poza{i}.txt")

        try:
            with open(annotation_path, 'w') as file:
                for annotation in annotations_list:
                    file.write(','.join(map(str, annotation[0])) + ',' + ','.join(map(str, annotation[1])) + '\n')
        except PermissionError:
            print(f"Permission denied: '{annotation_path}'. Please check the file permissions.")


def generate_image_dataset(background_path, jucarii_file_path, nonjucarii_file_path, num_images):
    generated_images = []
    generated_annotations = []

    for _ in range(num_images):
        img = Image.open(background_path)

        number_of_jucarii_images = random.randint(3, 5)
        number_of_nonjucarii_images = random.randint(3, 5)

        top_left_coordinates = (435, 59)
        bottom_right_coordinates = (1619, 934)

        jucarii_images = [os.path.join(jucarii_file_path, f) for f in os.listdir(jucarii_file_path) if
                          f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        nonjucarii_images = [os.path.join(nonjucarii_file_path, f) for f in os.listdir(nonjucarii_file_path) if
                             f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        selected_jucarii_images = random.sample(jucarii_images, number_of_jucarii_images)
        selected_nonjucarii_images = random.sample(nonjucarii_images, number_of_nonjucarii_images)

        occupied_coordinates = set()
        annotations_list = []

        for image_path in (selected_jucarii_images + selected_nonjucarii_images):
            image = Image.open(image_path)

            while True:
                position_x = random.uniform(0,
                                            1 - image.width / (bottom_right_coordinates[0] - top_left_coordinates[0]))
                position_y = random.uniform(0,
                                            1 - image.height / (bottom_right_coordinates[1] - top_left_coordinates[1]))

                position_x = int(
                    position_x * (bottom_right_coordinates[0] - top_left_coordinates[0]) + top_left_coordinates[0])
                position_y = int(
                    position_y * (bottom_right_coordinates[1] - top_left_coordinates[1]) + top_left_coordinates[1])

                # Check if the coordinates overlap with the ones already occupied
                new_area = set(
                    (x, y) for x in range(position_x, position_x + image.width) for y in
                    range(position_y, position_y + image.height)
                )

                if not new_area & occupied_coordinates:
                    occupied_coordinates.update(new_area)
                    break

            img.paste(image, (position_x, position_y))
            label = 'Jucarie' if image_path in selected_jucarii_images else 'Non-Jucarie'
            annotations_list.append(
                [position_x, position_y, position_x + image.width,
                 position_y + image.height, label]
            )

        # img_np = np.array(img) / 255.0
        generated_images.append(img)
        generated_annotations.append(annotations_list)
        global index
        index += 1
        print(index)

    return generated_images, generated_annotations
