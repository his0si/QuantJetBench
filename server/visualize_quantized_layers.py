import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 전략별 TFLite 모델 경로
strategy_models = {
    "Full INT8": "resnet_activation_int8_ptq.tflite",
    "Selective L2": "resnet_l2norm_qat_72percent_qat.tflite",
    "Selective Hessian": "resnet_hessian_qat_72percent_qat.tflite",
    "Selective Hybrid": "resnet_hybrid_qat_70percent_qat.tflite",
    "Selective Middle": "resnet_middle_qat_56percent_qat.tflite"
}

# 레이어 수 기준 리스트 (예: ResNet-50의 주요 conv+bn+relu 블록 중 일부 50개 추출 기준)
layer_names = [f"layer_{i}" for i in range(50)]

# 결과 저장 (전략 x 레이어 매트릭스)
quant_map = np.zeros((len(strategy_models), len(layer_names)))

def is_quantized_op(op):
    # TFLite에서 INT8 quantized 연산 확인 기준 (주로 int8 입력 + 출력)
    input_types = [t.dtype for t in op.inputs]
    output_types = [t.dtype for t in op.outputs]
    return all(t == tf.int8.as_datatype_enum for t in input_types + output_types)

for i, (strategy, model_path) in enumerate(strategy_models.items()):
    assert os.path.exists(model_path), f"{model_path} not found."
    
    # TFLite 모델 로딩
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    ops_details = interpreter._get_ops_details()

    quantized_layers = set()
    for op in ops_details:
        if is_quantized_op(op):
            layer_name = op['op_name']
            quantized_layers.add(layer_name)

    # 레이어 이름 기준으로 quant_map 채우기 (1: 양자화, 0: 비양자화)
    for j, lname in enumerate(layer_names):
        if lname in quantized_layers:
            quant_map[i, j] = 1

# 시각화
plt.figure(figsize=(20, 5))
sns.heatmap(quant_map, cmap='Blues', cbar=True, xticklabels=layer_names, yticklabels=strategy_models.keys())
plt.xlabel("ResNet Layer Index")
plt.ylabel("Quantization Strategy")
plt.title("Quantized Layer Map Across QAT Strategies (1=Quantized, 0=Not)")
plt.tight_layout()
plt.show()
