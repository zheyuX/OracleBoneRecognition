from zipfile import ZipFile
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# 解压数据集
train_zip_path = 'p4_train.zip'
test_zip_path = 'p4_test.zip'
train_extract_dir = 'p4_train'
test_extract_dir = 'p4_test'

with ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(train_extract_dir)

with ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(test_extract_dir)

# 列出解压后的文件
train_subdir = os.path.join(train_extract_dir, 'p4_train')
test_subdir = os.path.join(test_extract_dir, 'p4_test')
train_images = os.listdir(train_subdir)
test_images = os.listdir(test_subdir)

# 显示图像的函数
def display_images(image_paths, titles, num_images=3):
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    for i, image_path in enumerate(image_paths[:num_images]):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.show()

# 从测试集选择样本图像展示
sample_test_images = [os.path.join(test_subdir, img) for img in test_images[:3]]
display_images(sample_test_images, titles=['Test Image 1', 'Test Image 2', 'Test Image 3'])

# 预处理图像的函数
def preprocess_image(image, size=(64, 64)):
    image_resized = cv2.resize(image, size)
    _, image_binarized = cv2.threshold(image_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_normalized = image_binarized / 255.0
    return image_normalized

sample_train_images_dir = os.path.join(train_subdir, train_images[0])
sample_train_images = [os.path.join(sample_train_images_dir, img) for img in os.listdir(sample_train_images_dir)[:3]]
preprocessed_train_images = [preprocess_image(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)) for img_path in sample_train_images]

# 展示预处理后的图像
def display_preprocessed_images(images, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, img in enumerate(images):
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.show()

display_preprocessed_images(preprocessed_train_images, ['Preprocessed Image 1', 'Preprocessed Image 2', 'Preprocessed Image 3'])

# TensorFlow 模型构建
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 模型训练
model = build_model(10)  # 示例类别数量
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# PyTorch Faster R-CNN 模型构建
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 加载图像用于展示预测结果
def load_image(image_path):
    img = Image.open(image_path)
    return img

# 展示图像上的预测结果
def plot_predictions(image, predictions, threshold=0.5):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for element in range(len(predictions['boxes'])):
        box = predictions['boxes'][element].cpu().detach().numpy()
        score = predictions['scores'][element].cpu().detach().numpy()
        label = predictions['labels'][element].cpu().detach().numpy()
        if score > threshold:
            x, y, xmax, ymax = box
            rect = patches.Rectangle((x, y), xmax - x, ymax - y, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, f'{label}: {score:.2f}', fontsize=10, color='white', bbox=dict(facecolor='red', alpha=0.5))
    plt.show()

# 实际使用模型进行测试的函数
def test_model(model, test_image_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device).eval()
    img = load_image(test_image_path)
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)
    plot_predictions(img, prediction[0])
