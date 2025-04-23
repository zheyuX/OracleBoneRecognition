from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 加载图像
image_path_h02060 = '/1_Pre_test/h02060.jpg'
image_h02060 = Image.open(image_path_h02060)

# 调整图像大小
size = (256, 256)  # 示例大小，可根据实际需要调整
image_h02060_resized = image_h02060.resize(size)

# 灰度化处理
image_h02060_gray = image_h02060_resized.convert('L')

# 转换为 NumPy 数组以便后续处理
image_h02060_array = np.array(image_h02060_gray)

# 可视化处理后的图像
plt.figure(figsize=(6, 6))
plt.imshow(image_h02060_array, cmap='gray')
plt.title('Resized and Grayscale Image')
plt.axis('off')  # 不显示坐标轴
plt.show()

# 设置二值化阈值
thresh = 128
# 二值化处理
binary_image = cv2.threshold(image_h02060_array, thresh, 255, cv2.THRESH_BINARY)[1]
# 中值滤波
median_filtered = cv2.medianBlur(binary_image, 3)
# 锐化滤波，创建一个锐化核
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(median_filtered, -1, sharpen_kernel)
# Gabor滤波
gabor_kernel = cv2.getGaborKernel((5, 5), 4.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
gabor_filtered = cv2.filter2D(sharpened_image, cv2.CV_8UC3, gabor_kernel)
# 高斯滤波平滑图像
gaussian_blurred = cv2.GaussianBlur(sharpened_image, (3, 3), 0)

# 可视化处理效果
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 原始二值化图像
axes[0, 0].imshow(binary_image, cmap='gray')
axes[0, 0].set_title('Binary Image')
axes[0, 0].axis('off')
# 中值滤波后的图像
axes[0, 1].imshow(median_filtered, cmap='gray')
axes[0, 1].set_title('Median Filtered Image')
axes[0, 1].axis('off')
# 锐化后的图像
axes[0, 2].imshow(sharpened_image, cmap='gray')
axes[0, 2].set_title('Sharpened Image')
axes[0, 2].axis('off')
# Gabor滤波后的图像
axes[1, 0].imshow(gabor_filtered, cmap='gray')
axes[1, 0].set_title('Gabor Filtered Image')
axes[1, 0].axis('off')
# 高斯滤波后的图像
axes[1, 1].imshow(gaussian_blurred, cmap='gray')
axes[1, 1].set_title('Gaussian Blurred Image')
axes[1, 1].axis('off')
# 对比原始灰度图像和高斯滤波后的图像
axes[1, 2].imshow(image_h02060_array, cmap='gray')
axes[1, 2].set_title('Original Grayscale Image')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# 使用灰度图像
# 中值滤波去除噪点
median_filtered_gray = cv2.medianBlur(image_h02060_array, 3)
# 锐化滤波增强边缘
sharpened_image_gray = cv2.filter2D(median_filtered_gray, -1, sharpen_kernel)
# Gabor滤波突出特定方向的纹理
gabor_filtered_gray = cv2.filter2D(sharpened_image_gray, cv2.CV_8UC3, gabor_kernel)
# 高斯滤波平滑图像
gaussian_blurred_gray = cv2.GaussianBlur(sharpened_image_gray, (3, 3), 0)
# 直方图均衡化增强对比度
equalized_image = cv2.equalizeHist(image_h02060_array)

# 可视化处理效果
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
# 中值滤波后的灰度图像
axes[0, 0].imshow(median_filtered_gray, cmap='gray')
axes[0, 0].set_title('Median Filtered Gray Image')
axes[0, 0].axis('off')
# 锐化后的灰度图像
axes[0, 1].imshow(sharpened_image_gray, cmap='gray')
axes[0, 1].set_title('Sharpened Gray Image')
axes[0, 1].axis('off')
# Gabor滤波后的灰度图像
axes[1, 0].imshow(gabor_filtered_gray, cmap='gray')
axes[1, 0].set_title('Gabor Filtered Gray Image')
axes[1, 0].axis('off')
# 高斯滤波后的灰度图像
axes[1, 1].imshow(gaussian_blurred_gray, cmap='gray')
axes[1, 1].set_title('Gaussian Blurred Gray Image')
axes[1, 1].axis('off')
# 直方图均衡化后的灰度图像
axes[2, 0].imshow(equalized_image, cmap='gray')
axes[2, 0].set_title('Histogram Equalized Image')
axes[2, 0].axis('off')
# 原始灰度图像
axes[2, 1].imshow(image_h02060_array, cmap='gray')
axes[2, 1].set_title('Original Gray Image')
axes[2, 1].axis('off')

plt.tight_layout()
plt.show()


# Canny边缘检测
edges = cv2.Canny(image_h02060_array, 100, 200)
# 找到边缘检测后的连通组件
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 绘制边界框
image_with_contours = cv2.cvtColor(image_h02060_array, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)
# 可视化边缘检测结果
plt.figure(figsize=(8, 8))
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')
plt.show()
# 可视化绘制了边界框的图像
plt.figure(figsize=(8, 8))
plt.imshow(image_with_contours)
plt.title('Image with Contours')
plt.axis('off')
plt.show()


# 形态学操作
# 定义一个3x3的结构元素
kernel = np.ones((3,3), np.uint8)
# 腐蚀操作
erosion = cv2.erode(edges, kernel, iterations=1)
# 膨胀操作，对腐蚀后的图像进行膨胀，以突出文字部分
dilation = cv2.dilate(erosion, kernel, iterations=1)

# 可视化形态学操作的结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.tight_layout()
plt.show()