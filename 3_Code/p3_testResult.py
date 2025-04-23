import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zipfile import ZipFile

# 解压缩文件并处理图像的目录
zip_path = 'Figures.zip'
extract_dir = 'Figures'

# 如果目录不存在则创建
os.makedirs(extract_dir, exist_ok=True)

# 解压zip文件
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# 列出解压后的文件
extracted_files = os.listdir(extract_dir)
figures_folder = os.path.join(extract_dir, 'Figures')
figure_files = os.listdir(figures_folder)

# 函数：处理图像并检测边缘
def process_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)
    return img, img_contours, contours

# 处理第一张图像以示范
first_image_path = os.path.join(figures_folder, figure_files[0])
original_img, img_contours, contours = process_image(first_image_path)

# 展示原始图像与处理后的图像
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(original_img, cmap='gray')
ax[0].set_title('原始图像')
ax[0].axis('off')
ax[1].imshow(img_contours)
ax[1].set_title('带轮廓的图像')
ax[1].axis('off')
plt.show()

# 函数：优化图像处理以更好地检测轮廓
def optimize_image_processing(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    return img, img_contours, contours

# 优化第一张图像的处理
opt_original_img, opt_img_contours, opt_contours = optimize_image_processing(first_image_path)

# 展示优化后的原始图像和处理图像
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(opt_original_img, cmap='gray')
ax[0].set_title('原始图像')
ax[0].axis('off')
ax[1].imshow(opt_img_contours)
ax[1].set_title('优化后的带轮廓图像')
ax[1].axis('off')
plt.show()

# 函数：处理所有图像并提取轮廓
def process_all_images(image_folder):
    data = []
    for file in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file)
        _, img_contours, contours = optimize_image_processing(file_path)
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append([x, y, x + w, y + h, 1])  # [x1, y1, x2, y2, label]
        result_image_path = os.path.join(image_folder, 'contoured_' + file)
        cv2.imwrite(result_image_path, img_contours)
        data.append([file, bounding_boxes])
    return data

# 在文件夹中处理所有图像
image_data = process_all_images(figures_folder)

# 将数据保存到Excel文件
excel_path = 'Test_results.xlsx'
df = pd.DataFrame(image_data, columns=['图像名称', '标记'])
df.to_excel(excel_path, index=False)
