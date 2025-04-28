import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 완전 비활성화

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

# 1. Calibration 데이터 (작은 샘플로 충분)
(x_train, _), _ = cifar10.load_data()
x_train = x_train.astype(np.float32) / 255.0
calib_data = x_train[:100]

def representative_dataset():
    for i in range(calib_data.shape[0]):
        yield [calib_data[i:i+1]]

# 2. MPQ 변환만 수행
saved_model_dir = 'resnet_fp32_saved_model'

print("Converting to mixed-precision quantized TFLite (MPQ)...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # 가능한 연산은 quantize
    tf.lite.OpsSet.SELECT_TF_OPS     # 지원 안 되면 fallback
]

# MPQ TFLite 모델로 변환
tflite_mpq = converter.convert()

# 저장
output_path = 'resnet_custom_mpq.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_mpq)

size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"{output_path}: {size_mb:.2f} MB")
print("MPQ 모델 변환 완료!")
