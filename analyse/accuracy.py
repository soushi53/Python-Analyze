import tensorflow as tf
import numpy as np

# 🔁 モデル読み込み
model = tf.keras.models.load_model("C:\\Users\\soush\\Desktop\\GitHub\\agriScan\\analyse\\model\\vegetable_classifier_model_kind.keras")

# 📥 テストデータ読み込み（label_mode="int"が重要）
test_ds = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\soush\\Downloads\\Fruit-Images-Dataset-master\\Fruit-Images-Dataset-master\\Test",
    image_size=(180, 180),
    batch_size=32,
    label_mode="int",  # ← 数値ラベルとして取得（必要）
    shuffle=False       # ← 精度評価のために順番を固定
)

# 🔢 クラス名（フォルダ名から自動取得）
class_names = test_ds.class_names
print("クラス名（ラベル順）:", class_names)

# 📊 正解ラベルと予測結果を取得
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

# 📈 評価（正解率など）
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ モデルの正解率: {accuracy:.2%}\n")

print("📋 分類レポート:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

print("🧩 混同行列:\n")
print(confusion_matrix(y_true, y_pred))