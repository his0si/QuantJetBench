import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
import os

print("데이터 로딩 중...")
# Calibration을 위한 데이터 로딩
(x_train, _), (_, _) = cifar10.load_data()
x_train = x_train / 255.0

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

    print("\nEdgeTPU를 위한 INT8 모델 변환 중...")
    converter = tf.lite.TFLiteConverter.from_saved_model('resnet_fp32_saved_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_converter = True
    tflite_model_edgetpu = converter.convert()
    with open("resnet_full_int8_edgetpu.tflite", "wb") as f:
        f.write(tflite_model_edgetpu)
    print("EdgeTPU INT8 모델 변환 완료!")

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
for file in ['resnet_fp32.tflite', 'resnet_full_int8.tflite', 'resnet_full_int8_edgetpu.tflite', 'resnet_custom_mpq.tflite']:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"{file}: {size_mb:.2f} MB")
    else:
        print(f"{file}: 파일이 생성되지 않음")

print("\n모든 양자화 작업이 완료되었습니다!")
