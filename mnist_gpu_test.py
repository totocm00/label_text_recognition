# mnist_gpu_test.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 로그 깔끔히
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # (선택) oneDNN 비활성화 테스트시

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        details = tf.config.experimental.get_device_details(gpus[0])
        print("Using GPU:", details.get("device_name", "Unknown GPU"))
    except Exception as e:
        print("GPU config warning:", e)

# ====== 데이터: MNIST ======
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# [0,1] 정규화 + channel 차원 추가
x_train = (x_train / 255.0).astype("float32")[..., None]
x_test  = (x_test  / 255.0).astype("float32")[..., None]

batch_size = 256
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ====== 간단 CNN 모델 ======
model = keras.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ====== 학습 (2 epoch면 충분히 GPU 작동 확인 가능) ======
history = model.fit(train_ds, validation_data=test_ds, epochs=2)

# ====== 평가 ======
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"[DONE] Test accuracy: {test_acc:.4f}")
