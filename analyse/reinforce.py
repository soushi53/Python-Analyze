import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model

base_model = load_model("C:\\Users\\soush\\Desktop\\GitHub\\agriScan\\analyse\\model\\model_type1.keras")


# パスを指定

dataset_path = "C:\\Users\\soush\\Desktop\\dataset\\type\\2\\Training"

# データ読み込み
batch_size = 32
img_size = (180, 180)

train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="int"
)
num_classes = len(train_ds.class_names)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# クラス名（＝フォルダ名）
class_names = train_ds.class_names
print("分類対象の野菜ラベル：", class_names)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),        # 左右反転
    layers.RandomRotation(0.1),             # 回転 ±10%
    layers.RandomZoom(0.1),                 # ズーム ±10%
    layers.RandomContrast(0.1),             # コントラスト変化
])

model = tf.keras.Sequential([
    data_augmentation,                          # ← 追加
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,            # 通常の訓練データセット
    validation_data=val_ds,
    epochs=10
)

##step5
model.save("vegetable_classifier_model_kind.keras")


##step6
##import numpy as np
##from tensorflow.keras.utils import load_img, img_to_array

##img_path = 'C:\\Users\\soush\\Desktop\\GitHub\\agriScan\\analyse\\data\\test\\0001.jpg'
##img = load_img(img_path, target_size=(180, 180))
##img_array = img_to_array(img)
##img_array = tf.expand_dims(img_array, 0)  # バッチサイズを追加

##predictions = model.predict(img_array)
##predicted_class = class_names[np.argmax(predictions)]
##confidence = np.max(predictions)

##print(f"予測された野菜: {predicted_class}（確信度: {confidence:.2f}）")