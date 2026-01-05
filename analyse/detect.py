import tensorflow as tf

model = tf.keras.models.load_model("C:\\Users\\soush\\Desktop\\GitHub\\agriScan\\analyse\\model\\vegetable_classifier_model.keras")


import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

img_path = 'C:\\Users\\soush\\Desktop\\GitHub\\agriScan\\analyse\\data\\test\\0002.jpg'
img = load_img(img_path, target_size=(180, 180))
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # バッチサイズを追加

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"予測された野菜: {predicted_class}（確信度: {confidence:.2f}）")