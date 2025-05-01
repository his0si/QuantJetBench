import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.keras.datasets import cifar10

# Configuration
TEST_SIZE = 10000  # Increased: Number of test images to evaluate (10x)
NUM_RUNS = 100    # Number of runs for timing measurement (not used per-image timing)

def load_test_data():
    """Load and prepare test data from CIFAR-10."""
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    return x_test[:TEST_SIZE], y_test[:TEST_SIZE].flatten()

def get_model_size(path):
    """Get model size in MB."""
    if os.path.exists(path):
        if os.path.isdir(path):
            # Approximate directory size
            total = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total += os.path.getsize(fp)
        else:
            total = os.path.getsize(path)
        return total / (1024 * 1024)
    return 0.0

def evaluate_tflite_model(tflite_path, test_images, test_labels):
    """Evaluate TFLite model accuracy and inference time."""
    if not os.path.exists(tflite_path):
        print(f"Model not found: {tflite_path}")
        return None

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    times = []
    correct = 0
    for i in range(len(test_images)):
        img = test_images[i:i+1]
        if np.issubdtype(inp_det['dtype'], np.integer):
            scale, zp = inp_det['quantization']
            scale = scale or 1e-8
            inp = (img / scale + zp).astype(inp_det['dtype'])
        else:
            inp = img.astype(inp_det['dtype'])

        start = time.time()
        interpreter.set_tensor(inp_det['index'], inp)
        interpreter.invoke()
        times.append(time.time() - start)

        out = interpreter.get_tensor(out_det['index'])
        if np.issubdtype(out_det['dtype'], np.integer):
            s, zp = out_det['quantization']
            s = s or 1e-8
            out = (out.astype(np.float32) - zp) * s
        pred = np.argmax(out[0])
        if pred == test_labels[i]:
            correct += 1

    acc = correct / len(test_images)
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    size_mb = get_model_size(tflite_path)
    return acc, avg_time, fps, size_mb

def evaluate_keras_model(model_path, test_images, test_labels):
    """Evaluate Keras model accuracy and inference time."""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    model = tf.keras.models.load_model(model_path)
    times = []
    correct = 0
    for i in range(len(test_images)):
        img = test_images[i:i+1]
        start = time.time()
        pred = model.predict(img, verbose=0)
        times.append(time.time() - start)
        if np.argmax(pred[0]) == test_labels[i]:
            correct += 1

    acc = correct / len(test_images)
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    size_mb = get_model_size(model_path)
    return acc, avg_time, fps, size_mb

# Main
test_images, test_labels = load_test_data()
print(f"Loaded {len(test_images)} test images")

models = [
    ("FP32 Base Model", "resnet_fp32_saved_model", "keras"),
    ("Full INT8 PTQ", "resnet_activation_int8_ptq.tflite", "tflite"),
    ("Full INT8 QAT", "resnet_full_qat_100percent_qat.tflite", "tflite"),
    ("Selective L2 QAT", "resnet_l2norm_qat_72percent_qat.tflite", "tflite"),
    ("Selective Hessian QAT", "resnet_hessian_qat_72percent_qat.tflite", "tflite"),
    ("Selective Hybrid QAT", "resnet_hybrid_qat_70percent_qat.tflite", "tflite"),
    ("Selective Middle QAT", "resnet_middle_qat_56percent_qat.tflite", "tflite")
]

results = []
for name, path, mtype in models:
    print(f"\nEvaluating {name}...")
    if mtype == "tflite":
        out = evaluate_tflite_model(path, test_images, test_labels)
    else:
        out = evaluate_keras_model(path, test_images, test_labels)
    if out is None:
        print(f"  Skipped {name}")
        continue
    acc, avg_time, fps, size_mb = out
    results.append((name, acc*100, avg_time*1000, fps, size_mb))
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Avg inference time: {avg_time*1000:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  Model size: {size_mb:.2f} MB")

# Summary
print("\nSummary:")
print(f"{'Model':<25}{'Acc (%)':>10}{'Time (ms)':>15}{'FPS':>10}{'Size (MB)':>12}")
for name, acc, tms, fps, sz in results:
    print(f"{name:<25}{acc:10.2f}{tms:15.2f}{fps:10.1f}{sz:12.2f}")
