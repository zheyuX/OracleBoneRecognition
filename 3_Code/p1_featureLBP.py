import cv2
import numpy as np
from matplotlib import pyplot as plt

# 计算LBP图像的函数
def lbp_image(image):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取图像的尺寸
    height, width = gray_image.shape

    # 创建一个空的LBP图像，大小与输入图像相同
    lbp = np.zeros((height, width), np.uint8)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # 获取中心像素值
            center = gray_image[i][j]

            # 初始化要比较的像素
            pixels = []

            # 按顺时针方向从左上角像素开始，将邻近像素加入列表
            pixels.append(gray_image[i - 1][j - 1] > center)
            pixels.append(gray_image[i - 1][j] > center)
            pixels.append(gray_image[i - 1][j + 1] > center)
            pixels.append(gray_image[i][j + 1] > center)
            pixels.append(gray_image[i + 1][j + 1] > center)
            pixels.append(gray_image[i + 1][j] > center)
            pixels.append(gray_image[i + 1][j - 1] > center)
            pixels.append(gray_image[i][j - 1] > center)

            # 将布尔列表转换为一个字节，每个位表示一个邻近像素
            value = sum([1 << (7 - index) for index, val in enumerate(pixels) if val])

            # 将该字节值赋给LBP图像中与中心像素相同位置
            lbp[i, j] = value

    return lbp

# 加载图像，计算其LBP，并显示的函数
def process_and_display_images(filepaths):
    for filepath in filepaths:
        # 读取图像
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)

        # 计算图像的LBP
        lbp_img = lbp_image(image)

        # 显示LBP图像
        plt.imshow(lbp_img, cmap='gray')
        plt.show()

# 增强版计算LBP图像的函数
def lbp_image_enhanced(image):
    # 检查图像是否已经是灰度图
    if len(image.shape) > 2 and image.shape[2] == 3:
        # 将图像转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # 获取图像尺寸
    height, width = gray_image.shape

    # 创建一个空的LBP图像，大小与输入图像相同
    lbp = np.zeros((height, width), np.uint8)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # 获取中心像素值
            center = gray_image[i][j]

            # 初始化要比较的像素
            pixels = []

            # 按顺时针方向从左上角像素开始，将邻近像素加入列表
            pixels.append(gray_image[i - 1][j - 1] > center)
            pixels.append(gray_image[i - 1][j] > center)
            pixels.append(gray_image[i - 1][j + 1] > center)
            pixels.append(gray_image[i][j + 1] > center)
            pixels.append(gray_image[i + 1][j + 1] > center)
            pixels.append(gray_image[i + 1][j] > center)
            pixels.append(gray_image[i + 1][j - 1] > center)
            pixels.append(gray_image[i][j - 1] > center)

            # 将布尔列表转换为一个字节，每个位表示一个邻近像素
            value = sum([1 << (7 - index) for index, val in enumerate(pixels) if val])

            # 将该字节值赋给LBP图像中与中心像素相同位置
            lbp[i, j] = value

    return lbp

# 加载图像，计算其增强版LBP，并显示的函数
def process_and_display_images_enhanced(filepaths):
    # 定义轮廓线颜色（此例为蓝色）
    outline_color = (255, 0, 0)

    for filepath in filepaths:
        # 读取图像
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 假定图像已经是灰度图

        # 计算图像的LBP
        lbp_img = lbp_image_enhanced(image)

        # 在LBP图像上找到轮廓
        contours, _ = cv2.findContours(lbp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 在原始图像上绘制轮廓
        contour_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 将灰度图转换为BGR以便上色
        cv2.drawContours(contour_img, contours, -1, outline_color, 1)  # 用指定颜色绘制轮廓

        # 显示带有轮廓的图像
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.show()

# 已上传图像的文件路径
filepaths = ['h02060.jpg', 'w01637.jpg', 'w01870.jpg']
# 处理并显示LBP图像
process_and_display_images(filepaths)
# 处理并显示增强版LBP图像
process_and_display_images_enhanced(filepaths)
