# Intelligent Automated Object Sorting System

An intelligent robotic sorting system that combines **Computer Vision** and **Machine Learning** to detect, classify, and physically sort objects using a UR5 robotic arm. The system integrates two deep learning detection models (Faster R-CNN and YOLOv) with a depth camera and a robotic controller to enable fully automated, real-world object sorting.

---

## Project Structure

```
Intelligent-Automated-Object-Sorting-System/
├── FRCNN/                  # Faster R-CNN model weights and related files
├── YOLO/                   # YOLOv model weights and related files
├── Interfaces/
│   ├── robot_interface.py  # UR5 robot connection, movement, and gripper control
│   └── camera_interface.py # RealSense camera initialization, capture, and coordinate mapping
├── detection.py            # Object detection pipeline (model loading, inference, coordinate computation)
├── main.py                 # Entry point: orchestrates detection, positioning, and robot sorting
```

---

## Features

- **Dual Model Support**: Implements and compares two object detection architectures — Faster R-CNN (ResNet-50 FPN backbone) and YOLO — for classifying objects on a conveyor or workspace.
- **Depth-Aware Detection**: Uses an Intel RealSense depth camera to capture aligned RGB-D images and convert detected pixel positions to real-world millimeter coordinates.
- **Robotic Arm Integration**: Controls a UR5 robotic arm to physically pick up detected objects and move them to a designated drop position, based on computed 3D coordinates.
- **Custom Object Classes**: The models are trained to distinguish between domain-specific classes (e.g., "Jucarie" / "Non-Jucarie" — toy / non-toy), with adjustable detection thresholds.
- **Camera-to-Robot Coordinate Mapping**: Implements an offset calibration system to accurately translate camera frame coordinates into robot workspace coordinates.
- **Visual Feedback**: Renders bounding boxes and class labels on the captured frame using OpenCV for real-time inspection.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Madalin2605/Intelligent-Automated-Object-Sorting-System.git
cd Intelligent-Automated-Object-Sorting-System
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# On Linux/Mac
source .venv/bin/activate

# On Windows
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision opencv-python numpy pyrealsense2
```

> **Note:** Install the appropriate version of PyTorch for your system (CPU or CUDA) from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Place model weights

Put your trained model weights in the correct directories:

- Faster R-CNN weights → `FRCNN/model25.pth`
- YOLO weights → `YOLO/` (as appropriate)

### 5. Connect hardware

- Connect your **Intel RealSense** depth camera via USB.
- Connect your **UR5 robotic arm** over the network and ensure it is reachable.

---

## Usage

### Run the full sorting pipeline

```bash
python main.py
```

This will:
1. Connect to the UR5 robot and move it to the photo position.
2. Capture an aligned RGB-D image from the RealSense camera.
3. Run Faster R-CNN inference to detect and classify objects.
4. Convert detected pixel positions to robot workspace coordinates.
5. Command the robot to pick up each detected object and move it to the drop position.

### Run detection only (without robot)

```bash
python detection.py
```

Useful for testing the vision pipeline independently, without needing the robotic arm.

---

## Tech Stack

- **Python 3.9+**
- **PyTorch + TorchVision** — deep learning inference (Faster R-CNN, YOLO)
- **OpenCV** — image processing and visualization
- **Intel RealSense SDK (`pyrealsense2`)** — depth camera capture and 3D coordinate projection
- **NumPy** — numerical computation and coordinate transformations
- **UR5 Robot Interface** — custom interface for robot movement and gripper control
