# train_and_quantize.py
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import os

def build_resnet34(input_shape, num_classes):
    # 더 작은 모델 사용 (ResNet50 대신 ResNet18)
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg')
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model

print("데이터 로딩 중...")
# 데이터 로딩
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
input_shape = x_train.shape[1:]
num_classes = 10

print("모델 구축 및 학습 시작...")
# 모델 구축 및 학습
model = build_resnet34(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=0.1
)

history = model.fit(x_train, y_train,
          epochs=100,
          batch_size=32,
          validation_split=0.1,
          callbacks=[early_stop, reduce_lr])

print("\n모델 저장 중...")
# 모델 저장 (FP32)
model.save('resnet_fp32_saved_model')

print("\nTFLite 변환 시작...")
# Calibration을 위한 샘플 데이터 준비
print("Calibration 데이터 준비 중...")
calib_data = x_train[:50].astype(np.float32)

def representative_dataset():
    for i in range(50):
        yield [calib_data[i:i+1]]

try:
    print("\nFP32 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_saved_model('resnet_fp32_saved_model')
    tflite_model_fp32 = converter.convert()
    with open("resnet_fp32.tflite", "wb") as f:
        f.write(tflite_model_fp32)
    print("FP32 모델 변환 완료!")

    print("\nINT8 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_saved_model('resnet_fp32_saved_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model_quant = converter.convert()
    with open("resnet_full_int8.tflite", "wb") as f:
        f.write(tflite_model_quant)
    print("INT8 모델 변환 완료!")

    print("\nMPQ 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_saved_model('resnet_fp32_saved_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model_mpq = converter.convert()
    with open("resnet_custom_mpq.tflite", "wb") as f:
        f.write(tflite_model_mpq)
    print("MPQ 모델 변환 완료!")

except Exception as e:
    print(f"\n오류 발생: {str(e)}")

print("\n변환된 모델 파일 크기:")
for file in ['resnet_fp32.tflite', 'resnet_full_int8.tflite', 'resnet_custom_mpq.tflite']:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"{file}: {size_mb:.2f} MB")
    else:
        print(f"{file}: 파일이 생성되지 않음")

print("\n모든 작업이 완료되었습니다!")
