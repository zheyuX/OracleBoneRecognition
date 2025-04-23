import os
import numpy as np
import pandas as pd
import json
from PIL import Image, ImageDraw
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import precision_score, f1_score, jaccard_score, matthews_corrcoef
from sklearn.model_selection import KFold

# 创建遮罩图的函数
def create_mask(image_path, ann_data, image_size=(128, 128)):
    with Image.open(image_path) as img:
        img = img.resize(image_size)
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        for ann in ann_data:
            if ann[-1] == 1.0:  # 根据条件绘制矩形
                draw.rectangle([ann[0], ann[1], ann[2], ann[3]], fill=1)
        mask = np.array(mask)
    return np.array(img), mask.reshape(*mask.shape, 1)

# 构建 U-Net 模型的函数
def build_unet(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    u6 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(p1)
    u6 = concatenate([u6, c1])
    c6 = Conv2D(16, (3, 3), activation='relu', padding='same')(u6)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c6)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# 加载图像和标注数据的函数
def load_data(directory):
    images, masks = [], []
    for fname in os.listdir(directory):
        if fname.endswith('.json'):
            img_name = fname.split('.')[0] + '.jpg'
            img_path = os.path.join(directory, img_name)
            ann_path = os.path.join(directory, fname)
            with open(ann_path) as f:
                ann_data = json.load(f)['ann']
            img, mask = create_mask(img_path, ann_data)
            img = img_to_array(img)
            images.append(img)
            masks.append(mask)
    return np.array(images), np.array(masks)

# 交叉验证模型性能的函数
def cross_validate(images, masks, n_splits=5):
    kf = KFold(n_splits=n_splits)
    model_scores = []
    for train_idx, test_idx in kf.split(images):
        train_images, test_images = images[train_idx], images[test_idx]
        train_masks, test_masks = masks[train_idx], masks[test_idx]
        model = build_unet()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_images, train_masks, epochs=10, verbose=1)
        pred_masks = model.predict(test_images)
        pred_masks = (pred_masks > 0.5).astype(int)
        precision = precision_score(test_masks.flatten(), pred_masks.flatten())
        f1 = f1_score(test_masks.flatten(), pred_masks.flatten())
        dice = jaccard_score(test_masks.flatten(), pred_masks.flatten())
        matthews = matthews_corrcoef(test_masks.flatten(), pred_masks.flatten())
        mIoU = jaccard_score(test_masks.flatten(), pred_masks.flatten(), average='macro')
        mPA = np.mean(test_masks == pred_masks)
        model_scores.append([precision, f1, dice, matthews, mIoU, mPA])
    return np.mean(model_scores, axis=0)

# 在测试数据上应用模型并保存结果的函数
def apply_model_to_test(test_dir, model, output_file='Test_results.xlsx'):
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]
    results = []
    for image_path in test_images:
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        pred = (pred > 0.5).astype(int)
        results.append((os.path.basename(image_path), pred.flatten()))
    df = pd.DataFrame(results, columns=['Image', 'Predicted Mask'])
    df.to_excel(output_file, index=False)

# 主函数
def main():
    train_dir = 'Train/Train'
    test_dir = 'Test/Test'
    images, masks = load_data(train_dir)
    scores = cross_validate(images, masks)
    print('Model scores:', scores)
    model = build_unet()
    model.save_weights('model_weights.h5')
    model.load_weights('model_weights.h5')
    apply_model_to_test(test_dir, model)

if __name__ == '__main__':
    main()
