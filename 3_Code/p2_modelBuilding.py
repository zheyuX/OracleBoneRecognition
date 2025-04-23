import os
import numpy as np
import json
from PIL import Image, ImageDraw
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 构建 U-Net 模型的函数
def build_unet(input_shape):
    inputs = Input(input_shape)

    # 编码路径
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # 瓶颈层
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # 解码路径
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    # 输出层
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# 假设输入图像的形状为(128, 128, 1)
unet_model = build_unet((128, 128, 1))
unet_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 打印模型结构
print(unet_model.summary())

# 定义创建遮罩的函数
def create_mask(image_path, annotations, output_size):
    image = Image.open(image_path)
    image = image.resize(output_size)
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    for ann in annotations['ann']:
        if ann[-1] == 1.0:  # 检查标注最后一个字段，决定是否绘制
            rectangle = [tuple(ann[:2]), tuple(ann[2:4])]
            draw.rectangle(rectangle, fill=1)
    return np.array(image), np.array(mask)

# 加载示例图像和标注
example_image_path = os.path.join(train_extract_dir, 'b02519Z.jpg')
example_annotation_path = os.path.join(train_extract_dir, 'b02519Z.json')
with open(example_annotation_path, 'r') as f:
    example_annotations = json.load(f)

# 创建遮罩和处理图像
img, mask = create_mask(example_image_path, example_annotations, (128, 128))
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=-1)
mask = np.expand_dims(mask, axis=0)
mask = np.expand_dims(mask, axis=-1)

# 数据增强配置
data_gen_args = dict(rotation_range=15,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.05,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# 创建数据生成器
image_generator = image_datagen.flow(img, seed=1, batch_size=20)
mask_generator = mask_datagen.flow(mask, seed=1, batch_size=20)

# 将图像生成器和掩码生成器打包
train_generator = zip(image_generator, mask_generator)

# 训练模型
unet_model.fit(train_generator, steps_per_epoch=50, epochs=20)
