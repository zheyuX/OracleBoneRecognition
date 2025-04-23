import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A

# 定义训练数据集目录
train_extract_dir = 'Train/Train/'

# 定义示例图像和标注文件的路径
example_image_path = os.path.join(train_extract_dir, 'b02519Z.jpg')
example_annotation_path = os.path.join(train_extract_dir, 'b02519Z.json')

# 定义函数加载图像和标注
def load_image_and_annotations(img_path, ann_path):
    # 加载图像
    img = Image.open(img_path)
    # 加载标注
    with open(ann_path, 'r') as f:
        annotations = json.load(f)
    return img, annotations

# 加载示例图像和标注
example_image, example_annotations = load_image_and_annotations(example_image_path, example_annotation_path)

# 定义数据增强操作
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # 以50%的概率水平翻转
    A.Rotate(limit=15, p=0.5),  # 以50%的概率旋转±15度
    A.RandomScale(scale_limit=0.1, p=0.5)  # 以50%的概率随机缩放图像
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# 生成虚拟标签（所有标签都是1，因为都是有效的对象）
labels = [1] * len(example_annotations['ann'])

# 应用数据增强到图像及其标注
augmented = transform(image=np.array(example_image), bboxes=example_annotations['ann'], labels=labels)

# 展示增强后的图像和变换后的边界框
fig, ax = plt.subplots(1)
ax.imshow(augmented['image'])
for bbox in augmented['bboxes']:
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.show()

# 定义水平翻转图像的简单函数
def horizontal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

# 定义旋转图像的简单函数
def rotate_image(image, angle):
    return image.rotate(angle)

# 应用水平翻转
flipped_image = horizontal_flip(example_image)

# 应用旋转
rotated_image = rotate_image(example_image, 15)

# 展示原始图像及其变换后的版本
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(example_image)
ax[0].set_title('原始图像')
ax[0].axis('off')

ax[1].imshow(flipped_image)
ax[1].set_title('水平翻转')
ax[1].axis('off')

ax[2].imshow(rotated_image)
ax[2].set_title('旋转15度')
ax[2].axis('off')

plt.show()
