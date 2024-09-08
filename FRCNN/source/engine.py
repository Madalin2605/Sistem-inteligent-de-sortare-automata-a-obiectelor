import json
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from model import create_model
from utils import Averager
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader

plt.style.use('ggplot')


class Callback:
    def __init__(self):
        self.validation_losses = []

    def on_train_begin(self):
        pass

    def on_epoch_end(self, epoch, val_loss):
        self.validation_losses.append(val_loss)

    def on_train_end(self):
        with open('../outputs/validation_losses.json', 'w') as f:
            json.dump(self.validation_losses, f)


# Initialize IoU list
iou_values= []

# Function for running training iterations
def train(train_data_loader, model):
    print('Training')

    global train_itr
    global train_loss_list

    # Initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    # img_test = torch.zeros_like(torch.rand(3, 512, 512)) # IMAGINE DE TEST
    # plt.imshow(img_test.permute(1, 2, 0)) # PLOTARE IMAGINE DE TEST (AR TREBUI SA FIE NEAGRA!!!!!)
    # plt.show()

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        # print(f"Images: {images}")
        # print(f"Targets: {targets}")

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        # print(f"Images after list: {images}")
        # print(f"Targets after list: {targets}")

        loss_dict = model(images, targets)

        # Get model predictions
        model.eval()
        with torch.no_grad():
            predictions = model(images)
        model.train()

        # Calculate IoU for each prediction and corresponding target
        iou_epoch = []
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            target_boxes = target['boxes']

            iou = box_iou(pred_boxes, target_boxes)
            print(f"IOU shape: {iou.size()}")
            mean_iou = iou.max().cpu()
            iou_epoch.append(mean_iou)  # Assuming you want the mean IoU for each image

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1

        # Update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}, IoU: {np.mean(iou_epoch):.4f}")


    return train_loss_list, np.mean(iou_epoch)


# Function for running validation iterations
def validate(valid_data_loader, model):

    print('Validating')
    global val_itr
    global val_loss_list

    # Initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
            predictions = model(images)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)

        val_loss_hist.send(loss_value)

        val_itr += 1

        # Update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return val_loss_list

def moving_average(data, window_size):
    """Calculate the moving average of the given list."""
    return [np.mean(data[i - window_size:i]) for i in range(window_size, len(data) + 1)]

if __name__ == '__main__':

    # Initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # Get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # Define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # Initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1

    # Train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []

    # Name to save the trained model with
    MODEL_NAME = 'trained_after_epoch_'

    callback = Callback()

    # At the start of training
    callback.on_train_begin()

    # Start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")

        # Reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # Create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()

        # Start timer and carry out training and validation
        start = time.time()
        train_loss, iou_epoch = train(train_loader, model)
        val_loss, iou_epoch_valid = validate(valid_loader, model)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} mean IoU: {iou_epoch:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        # Store the mean IoU for this epoch
        iou_values.append(iou_epoch)

        # Save model after every 2 epochs
        if (epoch + 1) % SAVE_MODEL_EPOCH == 0:
            torch.save(model.state_dict(), f"{OUT_DIR}/{MODEL_NAME}{epoch + 1}.pth")
            print('SAVING MODEL COMPLETE...\n')

        # Save loss plots after 2 epochs
        if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch + 1}.png")
            print('SAVING PLOTS COMPLETE...')

        # Save loss plots and model once at the end
        if (epoch + 1) == NUM_EPOCHS:
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch + 1}.png")

            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch + 1}.pth")

        # At the end of each epoch
        callback.on_epoch_end(epoch, val_loss)

        plt.close('all')
        # Sleep for 5 seconds after each epoch
        time.sleep(5)

# Define the window size for the moving average
window_size = 5

# Assuming iou_values is the list of IoU values that you used to plot the graph
iou_values_smooth = moving_average(iou_values, window_size)

# After the training loop, plot the IoU values
plt.figure()
plt.plot(iou_values)
plt.xlabel('Epoch')
plt.ylabel('Mean IoU')
plt.title('Mean IoU over epochs for training')
plt.savefig(f"{OUT_DIR}/iou_over_epochs_train.png")

plt.figure()
plt.plot(iou_values_smooth)
plt.xlabel('Epoch')
plt.ylabel('Mean IoU')
plt.title('Mean IoU over epochs for training')
plt.savefig(f"{OUT_DIR}/iou_smooth_over_epochs_train.png")

