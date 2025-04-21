# coral_inference.py
import time
import os
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.adapters import classify
from PIL import Image

MODEL_PATHS = {
    'fp32': 'resnet_fp32.tflite',
    'full_int8': 'resnet_full_int8_edgetpu.tflite',  # Edge TPU 컴파일된 모델
    'custom_mpq': 'resnet_custom_mpq_edgetpu.tflite'  # Edge TPU 컴파일된 모델
}

def load_image():
    # 임의의 CIFAR-10 이미지처럼 32x32로 resize
    img = Image.new('RGB', (32, 32), (128, 128, 128))
    return img

def measure_inference(model_path, runs=100):
    # 파일 존재 여부 확인
    if not os.path.exists(model_path):
        print(f"Error: {model_path} 파일을 찾을 수 없습니다.")
        return

    try:
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        size = input_size(interpreter)

        img = load_image().resize(size)
        
        # 입력 타입에 따라 처리
        input_details = interpreter.get_input_details()
        if input_details[0]['dtype'] == np.uint8:
            input_tensor = np.asarray(img).astype(np.uint8)
        else:
            input_tensor = (np.asarray(img).astype(np.float32) / 255.0)
        
        input_tensor = input_tensor.reshape(1, size[0], size[1], 3)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()  # warmup

        start = time.time()
        for _ in range(runs):
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            _ = classify.get_classes(interpreter, top_k=1)
        end = time.time()

        avg_time = (end - start) / runs
        print(f"{model_path}: 평균 추론 시간 = {avg_time * 1000:.2f} ms")
    except Exception as e:
        print(f"Error processing {model_path}: {str(e)}")

# 현재 디렉토리의 모든 파일 출력
print("현재 디렉토리의 파일들:")
for file in os.listdir('.'):
    if file.endswith('.tflite'):
        print(f"- {file}")
print("\n=== 모델 테스트 시작 ===\n")

for name, path in MODEL_PATHS.items():
    print(f"=== {name} 모델 ===")
    measure_inference(path)
