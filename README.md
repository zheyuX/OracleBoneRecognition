# OracleBoneRecognition

This project aims to develop an automatic method for the segmentation and recognition of Oracle Bone Inscriptions (甲骨文). The challenge lies in processing images of these inscriptions, which often suffer from issues such as noise, unclear text, and distortions due to the materials used. The solution proposed uses deep learning techniques including U-Net for image segmentation and Faster R-CNN for text recognition.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Project Overview

Oracle Bone Inscriptions are ancient Chinese characters found on bones and turtle shells, primarily used for divination. These inscriptions are crucial for understanding ancient Chinese history, culture, and language. However, the quality of images from these inscriptions is often poor, affected by noise and distortions from the materials. This project addresses these challenges by combining advanced image preprocessing, segmentation, and recognition techniques using deep learning.

The key steps in this project include:
1. **Image Preprocessing**: Enhancement of image clarity and removal of noise using techniques like histogram equalization and Gabor filtering.
2. **Image Segmentation**: The use of the U-Net model for automatic segmentation of single-character inscriptions.
3. **Text Recognition**: Recognition of segmented characters using the Faster R-CNN model.

## Technologies Used

- **Deep Learning**: U-Net, Faster R-CNN
- **Image Processing**: Gabor Filtering, Canny Edge Detection, Local Binary Pattern (LBP)
- **Programming Languages**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib

## Model Architecture

### 1. U-Net for Image Segmentation
The U-Net model is used for segmenting the Oracle Bone Inscriptions into individual characters. It consists of an encoder-decoder architecture with skip connections, which helps preserve spatial details during the segmentation process.

### 2. Faster R-CNN for Text Recognition
After segmentation, the Faster R-CNN model is applied to the segmented characters to recognize the text. Faster R-CNN is a state-of-the-art object detection model that uses Region Proposal Networks (RPN) to propose regions of interest, followed by RoI pooling for classification and bounding box regression.

## Installation

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib

## Usage

### Step 1: Preprocess Oracle Bone Images
Ensure that the images are in the correct format (e.g., `.jpg`, `.png`). Use the preprocessing script to enhance the images:


### Step 2: Segment the Images
Run the segmentation model using U-Net to segment the characters:


### Step 3: Recognize Text
After segmentation, use the Faster R-CNN model to recognize the text in the segmented images:


### Step 4: Evaluate Results
The system provides a report on the accuracy of the segmentation and recognition processes.

## Results

The model achieved the following performance metrics:
- **Precision**: 87.5%
- **F1 Score**: 0.83
- **Dice Coefficient**: 0.68
- **MCC**: 0.55
- **MIoU**: 0.74
- **MPA**: 0.84

These results demonstrate the robustness of the model in accurately segmenting and recognizing Oracle Bone Inscriptions despite the challenges posed by noise and distortions.
