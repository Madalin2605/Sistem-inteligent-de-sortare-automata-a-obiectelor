from ultralytics.models.yolo import YOLO
import torch

if __name__ == '__main__':

    model = YOLO('yolov5n.pt')

    results = model.train(data='D:/YOLO/dataset_yolo/data.yaml', epochs=25, batch=4)