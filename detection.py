import time
import numpy as np
import cv2
import torch
import sys

from Interfaces.camera_interface import camera_interface as ci

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CLASSES = [
    'background', 'Jucarie', 'Non-Jucarie'
]

def calculate_center(boxes, classes):

    # AN EMPTY LIST
    center_coordinates = []

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box

        # CALCULATE THE CENTER COORDINATES
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        print("Center: ", center_x, center_y)

        # ADD CENTER COORDINATES
        center_coordinates.append([center_x, center_y])

    # AN EMPTY MAP
    result_map = {}

    # ADD THE RESULTS IN THE MAP LIST
    if len(classes) == len(center_coordinates):
        for class_, center in zip(classes, center_coordinates):
            if class_ not in result_map:
                result_map[class_] = []
            result_map[class_].append(center)
    else:
        print("Listele nu au aceeași lungime!")

    return result_map

def create_model(num_classes):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # freeze the model parameters
    # for param in model.parameters():
    #     param.requires_grad = False

    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print(f"Number of input features: {in_features}")

    # Define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def detect():
    # set the computation device
    device = torch.device('cpu')

    # load the model and the trained weights
    model = create_model(num_classes=3).to(device)
    model.load_state_dict(torch.load(
        'FRCNN/model25.pth', map_location=device
    ))
    model.eval()

    profile, pipeline, config = ci.init_camera()
    color_image, depth_frame, depth_scale, depth_intrinsics, color_intrinsics = ci.capture_aligned_images(pipeline, config, profile)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = 0.8

    # get the image file name for saving output later on
    orig_image = color_image.copy()
    color_image = np.transpose(color_image, (2, 0, 1)).astype(np.cfloat)
    color_image /= 255.0

    # convert to tensor
    color_image = torch.tensor(color_image, dtype=torch.float).to(device)

    # add batch dimension
    color_image = torch.unsqueeze(color_image, 0)
    with torch.no_grad():
        outputs = model(color_image)

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    # carry further only if there are detected boxes
    transformed_result = None

    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        # print("Boxes: ", boxes)
        scores = outputs[0]['scores'].data.numpy()

        # filter out boxes according to detection_threshold
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()

        # get all the predicited class names
        pred_classes = [CLASSES[i] for i, score in zip(outputs[0]['labels'].to(device).numpy(), scores) if score >= detection_threshold]

        # draw the bounding boxes and write the class name on top of it
        depths = []
        for j, box in enumerate(draw_boxes):
            xmin, ymin, xmax, ymax = box
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j],
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2, lineType=cv2.LINE_AA)

        result = calculate_center(boxes, pred_classes)
        print("result:", result)

        jucarii_results = []

        # for jucarie in result['Non-Jucarie']:
        #     jucarii_results.append([int(jucarie[0]), int(jucarie[1])])

        if 'Non-Jucarie' in result:
            for jucarie in result['Non-Jucarie']:
                jucarii_results.append([int(jucarie[0]), int(jucarie[1])])
        else:
            print("Key 'Jucarie' not found in the result.")
            sys.exit()

        transformed_result = ci.pixel_to_mm_coordinates(jucarii_results, depth_frame, depth_intrinsics)
        print("transformed_result:", transformed_result)

        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(0)

    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()

    return transformed_result

def compute_positions(photo_position, result, center_depth, offset_x_cam, offset_y_cam, offset_z_cam):
    position_with_offset = photo_position.copy()
    position_with_offset[0] = photo_position[0] + result[0] + offset_x_cam
    position_with_offset[1] = photo_position[1] - result[1] + offset_y_cam
    position_with_offset[2] = photo_position[2] - center_depth + offset_z_cam


    return position_with_offset
