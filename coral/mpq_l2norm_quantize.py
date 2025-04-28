
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

from tensorflow.keras.models import load_model

# 1. SavedModel → Keras 모델 로드
saved_model_dir = 'resnet_fp32_saved_model'
model = tf.keras.models.load_model(saved_model_dir)

# 2. L2 norm 기반 selective quantization
threshold = 100.0  # ← 실험하면서 조절 가능
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer

annotated_layers = []
for layer in model.layers:
    # Conv2D나 Dense에 대해서만 적용
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
        weights = layer.get_weights()
        if weights:
            l2_norm = np.linalg.norm(weights[0])
            print(f"{layer.name}: L2 norm = {l2_norm:.2f}")
            if l2_norm < threshold:
                annotated_layers.append(quantize_annotate_layer(layer))
                continue
    annotated_layers.append(layer)

# 3. 새로운 모델로 재조립
annotated_model = tf.keras.Sequential(annotated_layers)

# 4. Quantization 적용
with tfmot.quantization.keras.quantize_scope():
    quantized_model = tfmot.quantization.keras.quantize_apply(annotated_model)

# 5. 저장
quantized_saved_model_dir = 'resnet_l2norm_mpq_model'
quantized_model.save(quantized_saved_model_dir)

# 6. Calibration용 데이터 준비 (CIFAR-10 일부 샘플)
from tensorflow.keras.datasets import cifar10
(x_train, _), _ = cifar10.load_data()
x_train = x_train.astype(np.float32) / 255.0
calib_data = x_train[:100]

def representative_dataset():
    for i in range(calib_data.shape[0]):
        yield [calib_data[i:i+1]]

# 7. TFLite 변환 (Mixed Precision)
converter = tf.lite.TFLiteConverter.from_saved_model(quantized_saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
tflite_model.summary()

# 8. 저장
with open("resnet_custom_mpq_l2norm.tflite", "wb") as f:
    f.write(tflite_model)
size_mb = os.path.getsize("resnet_custom_mpq_l2norm.tflite") / (1024 * 1024)
print(f"\n완료: resnet_custom_mpq_l2norm.tflite 저장됨 ({size_mb:.2f} MB)")

