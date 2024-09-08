import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

    # freeze the model parameters
    # for param in model.parameters():
    #     param.requires_grad = False

    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print(f"Number of input features: {in_features}")

    # Define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == '__main__':
    model = create_model(2)
    print(model)
