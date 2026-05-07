import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ===================== 【本地路径】自动读取你的数据集 =====================
BASE_DIR = "./chest_xray"  # 本地相对路径，无需修改
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# 加载图片函数
def load_data(folder, img_size=(150, 150)):
    images = []
    labels = []
    # 0=正常, 1=肺炎
    for idx, cls in enumerate(["NORMAL", "PNEUMONIA"]):
        cls_path = os.path.join(folder, cls)
        for img in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img)
            image = load_img(img_path, target_size=img_size)
            image = img_to_array(image) / 255.0
            images.append(image)
            labels.append(idx)
    return np.array(images), np.array(labels)

# 加载数据
print("正在加载数据集...")
X_train, y_train = load_data(TRAIN_DIR)
X_test, y_test = load_data(TEST_DIR)

# 从训练集拆分 20% 作为验证集（作业要求）
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ===================== 构建CNN模型 =====================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===================== 训练模型 =====================
print("开始训练...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=32
)

# ===================== 保存训练曲线 =====================
os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.legend()
plt.savefig("figures/train_curve.png")
plt.close()

# ===================== 测试集评估 =====================
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== 测试集结果 =====")
print(f"准确率: {acc:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# 保存混淆矩阵
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("figures/confusion_matrix.png")
plt.close()

print("\n运行完成！图表保存在 figures 文件夹")