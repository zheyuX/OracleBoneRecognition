import zipfile
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 定义训练集和测试集的压缩文件路径及解压目录
train_zip_path = 'Train.zip'
test_zip_path = 'Test.zip'
train_extract_dir = 'Train/'
test_extract_dir = 'Test/'

# 解压训练集数据
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(train_extract_dir)

# 更新路径指向内层的Train文件夹
train_extract_dir = os.path.join(train_extract_dir, 'Train/')

# 列出训练文件夹内部的文件以确认
train_files = os.listdir(train_extract_dir)

# 定义加载图像及其注释的函数
def load_image_and_annotations(img_path, ann_path):
    # 加载图像
    img = Image.open(img_path)
    # 加载注释
    with open(ann_path, 'r') as f:
        annotations = json.load(f)
    return img, annotations

# 文件路径
example_image_path = os.path.join(train_extract_dir, 'b02519Z.jpg')
example_annotation_path = os.path.join(train_extract_dir, 'b02519Z.json')

# 加载图像和注释
example_image, example_annotations = load_image_and_annotations(example_image_path, example_annotation_path)

# 显示图像并绘制边界框
fig, ax = plt.subplots(1)
ax.imshow(example_image)
for ann in example_annotations['ann']:
    rect = patches.Rectangle((ann[0], ann[1]), ann[2] - ann[0], ann[3] - ann[1], linewidth=1, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)

plt.show()
