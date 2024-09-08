import torch

BATCH_SIZE = 4  # increase / decrease according to GPU memory
RESIZE_TO = 512  # resize the image for training and transforms
NUM_EPOCHS = 25  # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if torch.cuda.is_available():
    print("Using GPU...")
else:
    print("Using CPU...")

# Classes: 0 index is reserved for background
CLASSES = [
    'background', 'Jucarie', 'Non-Jucarie'
]
NUM_CLASSES = 3

# Whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2  # save model after these many epochs

