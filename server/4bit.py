# -------------------------
# 4-bit Quantization Full Pipeline
# -------------------------

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.datasets import cifar10
import os
import shutil

# Check tfmot version
print("TFMOT version:", tfmot.__version__)

# --- Configuration ---
FINETUNE_SIZE = 45000
CALIB_SIZE = 5000
TEST_SIZE = 10000
QAT_EPOCHS = 20
QAT_LEARNING_RATE = 1e-4
TARGET_QUANT_PERCENTAGE = 0.70
MIDDLE_EXCLUDE_PERCENT = 0.15

saved_model_dir = 'resnet_fp32_saved_model'

# --- Load Model ---
if not os.path.exists(saved_model_dir):
    print("No saved model found. Creating a dummy model.")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.save(saved_model_dir)
else:
    model = tf.keras.models.load_model(saved_model_dir)

# Dummy predict to make sure model is built
_ = model(tf.random.uniform((1, 32, 32, 3)))

# --- Load CIFAR10 ---
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Shuffle
np.random.seed(42)
shuffle_idx = np.random.permutation(len(x_train))
x_train, y_train = x_train[shuffle_idx], y_train[shuffle_idx]

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

finetune_data = x_train[:FINETUNE_SIZE]
finetune_labels = y_train[:FINETUNE_SIZE].flatten()
calib_data = x_train[FINETUNE_SIZE:FINETUNE_SIZE+CALIB_SIZE]
calib_labels = y_train[FINETUNE_SIZE:FINETUNE_SIZE+CALIB_SIZE].flatten()

# --- Representative Dataset ---
def representative_dataset():
    for i in range(len(calib_data)):
        yield [calib_data[i:i+1]]

# --- Full 4-bit QAT Preparation ---
def prepare_qat_4bit(model):
    with tfmot.quantization.keras.quantize_scope():
        scheme = tfmot.quantization.keras.experimental.default_4bit_quantize_scheme.Default4BitQuantizeScheme()
        config = tfmot.quantization.keras.experimental.QuantizeConfig(quantize_scheme=scheme)
        qat_model = tfmot.quantization.keras.quantize_model(model, quantize_config=config)
    return qat_model

# --- Fine-tune QAT Model ---
def finetune_qat_model(model, save_path):
    model.compile(optimizer=tf.keras.optimizers.Adam(QAT_LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(finetune_data, finetune_labels, epochs=QAT_EPOCHS, batch_size=32,
              validation_data=(x_test, y_test), verbose=1)
    model.save(save_path)

# --- TFLite Conversion for 4-bit ---
def convert_to_tflite_4bit(saved_model_path, output_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    try:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_INT4
        ]
    except AttributeError:
        print("Warning: 4-bit OpsSet not available in your TensorFlow version. Trying INT8 fallback.")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite 4-bit model saved to {output_path}")

# --- Evaluate TFLite Model ---
def evaluate_tflite_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    for i in range(len(x_test)):
        img = x_test[i:i+1]
        label = y_test[i]

        scale, zero_point = input_details[0]['quantization']
        if scale == 0: scale = 1e-8
        img = (img / scale + zero_point).astype(input_details[0]['dtype'])

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if output_details[0]['dtype'] in [np.int8, np.uint8]:
            scale_out, zero_point_out = output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point_out) * scale_out

        pred = np.argmax(output)
        if pred == label:
            correct += 1

        if (i+1) % 1000 == 0:
            print(f"Evaluated {i+1}/{len(x_test)} samples...")

    acc = correct / len(x_test)
    print(f"TFLite model accuracy: {acc*100:.2f}% ({correct}/{len(x_test)})")
    return acc

# --- Main Run ---
if __name__ == "__main__":
    # Clone model to apply QAT
    model_for_qat = tf.keras.models.clone_model(model)
    model_for_qat.set_weights(model.get_weights())

    # 1. Prepare 4bit QAT Model
    qat_model = prepare_qat_4bit(model_for_qat)

    # 2. Fine-tune
    qat_save_path = "resnet_qat_4bit_saved_model"
    finetune_qat_model(qat_model, qat_save_path)

    # 3. Convert to 4bit TFLite
    tflite_save_path = "resnet_qat_4bit.tflite"
    convert_to_tflite_4bit(qat_save_path, tflite_save_path)

    # 4. Evaluate TFLite model
    evaluate_tflite_model(tflite_save_path)

print("\nAll done!")

