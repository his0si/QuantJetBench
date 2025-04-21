# train_and_quantize.py
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def build_resnet34(input_shape, num_classes):
    # 더 작은 모델 사용 (ResNet50 대신 ResNet18)
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg')  # GlobalAveragePooling2D를 직접 사용하지 않고 pooling 옵션으로 대체
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model

# 데이터 로딩
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
input_shape = x_train.shape[1:]
num_classes = 10

# 모델 구축 및 학습
model = build_resnet34(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 메모리 사용량을 줄이기 위한 콜백 추가
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,  # 10 에포크 동안 개선이 없으면 중단
    restore_best_weights=True,
    verbose=1
)

model.fit(x_train, y_train,
          epochs=100,  # 에포크 수를 100으로 설정
          batch_size=32,
          validation_split=0.1,
          callbacks=[early_stop, reduce_lr])

# 모델 저장 (FP32)
model.save('resnet_fp32_saved_model')

# ================== 양자화 (Full INT8) ==================
# Calibration을 위한 샘플 데이터 (더 작은 크기로)
calib_data = x_train[:50].astype(np.float32)

# full int8 양자화
def representative_dataset():
    for i in range(50):  # 샘플 수를 줄임
        yield [calib_data[i:i+1]]

converter = tf.lite.TFLiteConverter.from_saved_model('resnet_fp32_saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()

with open("resnet_full_int8.tflite", "wb") as f:
    f.write(tflite_model_quant)

# ================== 네 방법 (예시 MPQ 구현) ==================
converter = tf.lite.TFLiteConverter.from_saved_model('resnet_fp32_saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model_mpq = converter.convert()

with open("resnet_custom_mpq.tflite", "wb") as f:
    f.write(tflite_model_mpq)

# FP32 모델도 TFLite 변환
converter = tf.lite.TFLiteConverter.from_saved_model('resnet_fp32_saved_model')
tflite_model_fp32 = converter.convert()
with open("resnet_fp32.tflite", "wb") as f:
    f.write(tflite_model_fp32)
