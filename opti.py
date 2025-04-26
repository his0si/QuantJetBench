import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.datasets import cifar10
import os # <-- Import os

# Load saved FP32 model
saved_model_dir = 'resnet_fp32_saved_model'
# Ensure the model is loaded within the quantize_scope if it was previously quantized
# For a standard FP32 model, this isn't strictly necessary, but good practice if unsure.
# with tfmot.quantization.keras.quantize_scope():
#    base_model = tf.keras.models.load_model(saved_model_dir)
# Let's assume it's a standard FP32 Keras model for now
try:
    base_model = tf.keras.models.load_model(saved_model_dir)
    print("Successfully loaded base model.")
    base_model.summary() # Good idea to check the loaded model structure
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'resnet_fp32_saved_model' exists and is a valid Keras SavedModel.")
    exit() # Exit if model loading fails

# Prepare calibration and test data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Reduce dataset size for faster testing/debugging if needed
CALIB_SIZE = 100
TEST_SIZE = 1000
calib_data = x_train[:CALIB_SIZE]
calib_labels = y_train[:CALIB_SIZE].flatten() # Needed for Hessian/Hybrid sensitivity
test_images = x_test[:TEST_SIZE]
test_labels = y_test[:TEST_SIZE].flatten()

print(f"Using {len(calib_data)} calibration images and {len(test_images)} test images.")

def representative_dataset():
    # Ensure dataset shape matches model input
    # Assuming input shape is (None, height, width, channels)
    # E.g., for CIFAR-10 with ResNet typically (None, 32, 32, 3)
    # Adjust if your model expects a different shape
    for i in range(len(calib_data)):
        yield [calib_data[i:i+1]]

# Utility: convert annotated model to TFLite
def convert_to_tflite(saved_model_path, output_path, mixed_precision=False):
    # Check if the saved model directory exists
    if not os.path.isdir(saved_model_path):
         print(f"Error: Saved model directory not found: {saved_model_path}")
         return False # Indicate failure

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset

        if mixed_precision:
            # For mixed precision (some ops INT8, some float32)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TFLite built-ins
                tf.lite.OpsSet.SELECT_TF_OPS  # Enable TF ops if needed (though ideally avoided)
            ]
            # Set input/output types to float for mixed precision usually
            # Or keep them as integer if that's specifically desired for the quantized parts
            # Let's assume float input/output for flexibility with mixed precision
            converter.inference_input_type = tf.float32 # Or tf.uint8 if input is int quantized
            converter.inference_output_type = tf.float32 # Or tf.uint8 if output is int quantized

        else:
            # For full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8 # Typically uint8 for INT8 models
            converter.inference_output_type = tf.uint8 # Typically uint8 for INT8 models

        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        # Check size after writing
        file_size_mb = os.path.getsize(output_path)/(1024*1024) if os.path.exists(output_path) else 0
        print(f"Saved TFLite model: {output_path} ({file_size_mb:.2f} MB)")
        return True # Indicate success

    except Exception as e:
        print(f"Error during TFLite conversion for {output_path}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for conversion errors
        return False # Indicate failure


# 1) Weight L2-norm based selective quantization
def quantize_weight_l2(model, threshold=100.0):
    print("\n--- Starting L2 Norm Quantization ---")
    QuantAnnotate = tfmot.quantization.keras.quantize_annotate_layer

    def apply_quantization_to_layer(layer):
        # Check if it's a layer type we want to potentially quantize
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights = layer.get_weights()
            # Ensure the layer has weights (kernel is usually the first weight)
            if weights and len(weights) > 0 and weights[0] is not None:
                try:
                    # Calculate L2 norm of the kernel weights
                    l2_norm = np.linalg.norm(weights[0].flatten()) # Flatten just in case
                    print(f"Layer {layer.name}: L2 Norm = {l2_norm:.4f}")
                    if l2_norm < threshold:
                        print(f"  -> Annotating {layer.name} for quantization (L2 < {threshold})")
                        return QuantAnnotate(layer)
                    else:
                         print(f"  -> Skipping {layer.name} (L2 >= {threshold})")
                except Exception as e:
                    print(f"Could not process layer {layer.name}: {e}")
        # Return the original layer if not quantizing or not applicable
        return layer

    # Clone the model, applying the annotation function to each layer
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_quantization_to_layer,
    )

    # Apply quantization wrappers based on the annotations
    # Use quantize_scope to use QuantizeAwareConv2D etc.
    with tfmot.quantization.keras.quantize_scope():
        # `quantize_apply` creates the modifications needed for QAT
        quant_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # Compile the model AFTER applying quantization. A dummy compile is often sufficient
    # if only saving/converting, but using a real optimizer/loss is safer.
    quant_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Compiled quantized model (L2 Norm).")

    # Save the Keras model (needed for TFLiteConverter.from_saved_model)
    save_path = 'model_l2norm_mpq'
    try:
        quant_model.save(save_path)
        print(f"Saved annotated Keras model: {save_path}")
        # Convert to TFLite
        convert_to_tflite(save_path, 'resnet_l2norm_mpq.tflite', mixed_precision=True)
    except Exception as e:
        print(f"Error saving or converting model {save_path}: {e}")


# 2) Activation-based full INT8 quantization (Post-Training Quantization)
def quantize_activation(model):
    print("\n--- Starting Full INT8 Post-Training Quantization ---")
    # This method uses TFLiteConverter directly for PTQ, doesn't modify the Keras model with TF-MOT wrappers
    save_path = 'model_activation_int8_keras_temp' # Temporary save of original model
    try:
        model.save(save_path)
        print(f"Saved temporary Keras model: {save_path}")
        # Convert directly using TFLiteConverter settings for full INT8 PTQ
        convert_to_tflite(save_path, 'resnet_activation_int8.tflite', mixed_precision=False)
        # Clean up temporary Keras model directory
        # Be careful with shutil.rmtree!
        # import shutil
        # if os.path.isdir(save_path):
        #     shutil.rmtree(save_path)
        #     print(f"Removed temporary directory: {save_path}")

    except Exception as e:
        print(f"Error saving or converting model for activation quantization: {e}")


# 3) Hessian-based sensitivity (approximate with gradients)
def quantize_hessian(model, threshold=1e-3):
    print("\n--- Starting Hessian (Gradient Approximation) Quantization ---")
    # Compute per-layer sensitivity: mean squared gradient * weight magnitude
    sensitivity = {}
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Ensure labels match model output (sparse vs categorical) and data shapes are correct
    sample_images = calib_data # Use a representative subset
    sample_labels = calib_labels # Use corresponding labels

    try:
        with tf.GradientTape() as tape:
            # Ensure model is callable and shapes match
            print(f"Calculating gradients with input shape: {sample_images.shape}")
            preds = model(sample_images, training=False) # Ensure training=False
            loss = loss_fn(sample_labels, preds)
            print(f"Calculated loss: {loss.numpy()}")

        # Get trainable variables *before* applying quantization wrappers
        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        if grads is None or not any(g is not None for g in grads):
             print("Warning: Gradients are None. Check model, loss function, and data.")
             return # Cannot proceed without gradients

        print(f"Calculated {len(grads)} gradients for {len(trainable_vars)} variables.")

        # Map variable to layer and calculate sensitivity score
        layer_sens = {}
        for var, grad in zip(trainable_vars, grads):
            if grad is None: # Skip if gradient couldn't be computed for a variable
                print(f"Warning: Gradient is None for variable {var.name}")
                continue
            # Attempt to map variable name back to layer name (this can be fragile)
            # Assumes variable names like 'conv2d_3/kernel:0'
            layer_name_parts = var.name.split('/')
            if len(layer_name_parts) > 1:
                layer_name = layer_name_parts[0]
                # Approximate sensitivity: Mean of (Grad * Weight)^2
                score = tf.reduce_mean(tf.square(grad * var)).numpy()
                layer_sens[layer_name] = layer_sens.get(layer_name, 0) + score
                print(f"Layer {layer_name} (from var {var.name}): Sensitivity Score += {score:.6f}")
            else:
                 print(f"Warning: Could not determine layer name from variable {var.name}")

        max_sens = max(layer_sens.values()) if layer_sens else 1.0
        print(f"Max sensitivity score: {max_sens:.6f}")

        QuantAnnotate = tfmot.quantization.keras.quantize_annotate_layer

        def apply_quantization_to_layer_hessian(layer):
            s = layer_sens.get(layer.name, 0)
            # Quantize layers with LOW sensitivity score
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)) and s < threshold:
                print(f"  -> Annotating {layer.name} for quantization (Sensitivity {s:.6f} < {threshold})")
                return QuantAnnotate(layer)
            elif layer.name in layer_sens:
                 print(f"  -> Skipping {layer.name} (Sensitivity {s:.6f} >= {threshold})")
            return layer

        # Clone the model applying the annotation function
        annotated_model = tf.keras.models.clone_model(
            model,
            clone_function=apply_quantization_to_layer_hessian,
        )

        # Apply quantization wrappers
        with tfmot.quantization.keras.quantize_scope():
            quant_model = tfmot.quantization.keras.quantize_apply(annotated_model)

        # Compile
        quant_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Compiled quantized model (Hessian).")

        # Save and Convert
        save_path = 'model_hessian_mpq'
        quant_model.save(save_path)
        print(f"Saved annotated Keras model: {save_path}")
        convert_to_tflite(save_path, 'resnet_hessian_mpq.tflite', mixed_precision=True)

    except Exception as e:
        print(f"Error during Hessian quantization: {e}")
        import traceback
        traceback.print_exc()


# 4) Hybrid: combine L2 norm + activation range + sensitivity
def quantize_hybrid(model, alpha=0.4, beta=0.4, gamma=0.2, threshold=0.5):
    print("\n--- Starting Hybrid Quantization ---")
    # --- Calculate L2 norms ---
    l2_norms = {}
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
             weights = layer.get_weights()
             if weights and len(weights) > 0 and weights[0] is not None:
                 l2_norms[layer.name] = np.linalg.norm(weights[0].flatten())
    max_l2 = max(l2_norms.values()) if l2_norms else 1.0
    print(f"Calculated L2 norms for {len(l2_norms)} layers. Max L2: {max_l2:.4f}")

    # --- Calculate activation ranges ---
    act_min, act_max = {}, {}
    try:
        # Need to create a model that outputs all intermediate activations
        # Ensure Input shape matches the model's expectation
        inp = tf.keras.Input(shape=model.input_shape[1:], name="hybrid_input")
        # Create a model that returns outputs of all layers
        layer_outputs = [layer.output for layer in model.layers]
        intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs)

        # Run calibration data through the intermediate model
        print(f"Calculating activations with input shape: {calib_data.shape}")
        activations = intermediate_model.predict(calib_data, verbose=0)

        range_ = {}
        for layer, act in zip(model.layers, activations):
             # Check if activation is numeric (skip inputs, etc.)
             if isinstance(act, np.ndarray) and np.issubdtype(act.dtype, np.number):
                 act_min[layer.name] = np.min(act)
                 act_max[layer.name] = np.max(act)
                 range_[layer.name] = act_max[layer.name] - act_min[layer.name]
             else:
                 # Handle non-numeric or unexpected outputs if necessary
                 pass # print(f"Skipping activation range for layer {layer.name} - Output type: {type(act)}")


        max_range = max(range_.values()) if range_ else 1.0
        print(f"Calculated activation ranges for {len(range_)} layers. Max Range: {max_range:.4f}")

    except Exception as e:
        print(f"Error calculating activation ranges: {e}")
        import traceback
        traceback.print_exc()
        # Proceed without activation ranges if calculation fails
        range_ = {}
        max_range = 1.0


    # --- Calculate sensitivity (reuse gradient approach) ---
    layer_sens = {}
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    sample_images = calib_data # Use a representative subset
    sample_labels = calib_labels # Use corresponding labels

    try:
        with tf.GradientTape() as tape:
            preds = model(sample_images, training=False)
            loss = loss_fn(sample_labels, preds)

        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        if grads is None or not any(g is not None for g in grads):
             print("Warning: Gradients are None for sensitivity calculation.")
        else:
            for var, grad in zip(trainable_vars, grads):
                if grad is None: continue
                layer_name_parts = var.name.split('/')
                if len(layer_name_parts) > 1:
                    layer_name = layer_name_parts[0]
                    score = tf.reduce_mean(tf.square(grad * var)).numpy()
                    layer_sens[layer_name] = layer_sens.get(layer_name, 0) + score

        max_sens = max(layer_sens.values()) if layer_sens else 1.0
        print(f"Calculated sensitivity for {len(layer_sens)} layers. Max Sensitivity: {max_sens:.6f}")

    except Exception as e:
        print(f"Error during sensitivity calculation for hybrid: {e}")
        layer_sens = {}
        max_sens = 1.0

    # --- Compute hybrid score and annotate ---
    QuantAnnotate = tfmot.quantization.keras.quantize_annotate_layer

    def apply_quantization_to_layer_hybrid(layer):
        # Only consider layers we can quantize and have scores for
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            score_l2 = alpha * (l2_norms.get(layer.name, 0) / max_l2) if max_l2 != 0 else 0
            score_range = beta * (range_.get(layer.name, 0) / max_range) if max_range != 0 else 0
            score_sens = gamma * (layer_sens.get(layer.name, 0) / max_sens) if max_sens != 0 else 0
            hybrid_score = score_l2 + score_range + score_sens

            print(f"Layer {layer.name}: L2={score_l2:.3f}, Range={score_range:.3f}, Sens={score_sens:.3f} -> Hybrid Score = {hybrid_score:.4f}")

            # Quantize if score is BELOW threshold (assuming lower score = less impact = safer to quantize)
            if hybrid_score < threshold:
                print(f"  -> Annotating {layer.name} for quantization (Score < {threshold})")
                return QuantAnnotate(layer)
            else:
                print(f"  -> Skipping {layer.name} (Score >= {threshold})")

        return layer # Return original layer otherwise

    # Clone the model applying the annotation function
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_quantization_to_layer_hybrid,
    )

    # Apply quantization wrappers
    with tfmot.quantization.keras.quantize_scope():
        quant_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # Compile
    quant_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Compiled quantized model (Hybrid).")

    # Save and Convert
    save_path = 'model_hybrid_mpq'
    try:
        quant_model.save(save_path)
        print(f"Saved annotated Keras model: {save_path}")
        convert_to_tflite(save_path, 'resnet_hybrid_mpq.tflite', mixed_precision=True)
    except Exception as e:
        print(f"Error saving or converting hybrid model: {e}")


# Evaluate TFLite model
def evaluate_tflite(tflite_path, test_images, test_labels):
    if not os.path.exists(tflite_path):
        print(f"Skipping evaluation: TFLite file not found: {tflite_path}")
        return

    print(f"\n--- Evaluating {os.path.basename(tflite_path)} ---")
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        # Check input type
        input_is_int = np.issubdtype(input_details['dtype'], np.integer)
        print(f"Model Input Type: {input_details['dtype']}")

        correct = 0
        num_eval_images = len(test_images)
        for i in range(num_eval_images):
            img = test_images[i]
            label = test_labels[i]

            # Prepare input based on model's expected type
            if input_is_int:
                # Scale to UINT8 range and cast
                # Check quantization parameters if available for precise scaling
                scale, zero_point = input_details['quantization']
                if scale == 0 and zero_point == 0: # Default for float fallback models or unquantized inputs
                     # Fallback: Assume standard 0-255 mapping if no quant params
                     inp = np.expand_dims((img * 255.0).astype(input_details['dtype']), axis=0)
                     print("Warning: Input is integer type but quantization params are 0. Using simple 0-255 scaling.", end='\r')
                else:
                    inp = np.expand_dims((img / scale + zero_point).astype(input_details['dtype']), axis=0)

            else:
                # Assume float input
                inp = np.expand_dims(img.astype(input_details['dtype']), axis=0)

            # Ensure input shape matches model's expected input shape
            # This is crucial if the model expects e.g., (1, H, W, C)
            interpreter.set_tensor(input_details['index'], inp)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details['index'])

            # Handle output based on its type
            output_is_int = np.issubdtype(output_details['dtype'], np.integer)
            if output_is_int:
                # Dequantize output if necessary for comparison
                scale, zero_point = output_details['quantization']
                if not (scale == 0 and zero_point == 0): # Check if quantized
                     dequantized_output = scale * (output_data.astype(np.float32) - zero_point)
                     pred = np.argmax(dequantized_output[0])
                else: # Integer output but not quantized? Use argmax directly.
                     pred = np.argmax(output_data[0])
            else:
                 # Float output
                 pred = np.argmax(output_data[0])

            if pred == label:
                correct += 1

            # Print progress
            print(f"Evaluating: {i+1}/{num_eval_images}", end='\r')

        accuracy = correct / num_eval_images
        print(f"\n{os.path.basename(tflite_path)} -> Accuracy: {accuracy*100:.2f}% ({correct}/{num_eval_images})")

    except Exception as e:
        print(f"Error evaluating {tflite_path}: {e}")
        import traceback
        traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the base model is loaded successfully before proceeding
    if 'base_model' in locals() or 'base_model' in globals():

        # Run selective quantization methods (using TF-MOT wrappers)
        #quantize_weight_l2(base_model, threshold=100.0)
        #quantize_hessian(base_model, threshold=1e-3) # Adjust threshold as needed
        quantize_hybrid(base_model, alpha=0.4, beta=0.4, gamma=0.2, threshold=0.5) # Adjust weights/threshold

        # Run full INT8 post-training quantization (using TFLiteConverter directly)
        quantize_activation(base_model)

        # Evaluate each resulting TFLite model
        tflite_files_to_evaluate = [
            'resnet_l2norm_mpq.tflite',
            'resnet_activation_int8.tflite',
            'resnet_hessian_mpq.tflite',
            'resnet_hybrid_mpq.tflite'
        ]
        for model_file in tflite_files_to_evaluate:
            evaluate_tflite(model_file, test_images, test_labels)

    else:
        print("Base model was not loaded. Cannot proceed with quantization.")