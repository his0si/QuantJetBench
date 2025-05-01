import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.datasets import cifar10
import os
import shutil

# --- Configuration ---
FINETUNE_SIZE = 45000 # Increased: Use 10000 images for QAT fine-tuning (closer to full CIFAR)
CALIB_SIZE = 5000     # Increased: Use 1000 images for PTQ and post-QAT TFLite calibration
TEST_SIZE = 10000      # Keep test size reasonable for quick evaluation
QAT_EPOCHS = 5       # Increased: Number of epochs for QAT fine-tuning (suggested >= 20)
QAT_LEARNING_RATE = 1e-4 # Reduced: Lower learning rate for QAT fine-tuning (suggested 1e-4 or 1e-5)
TARGET_QUANT_PERCENTAGE = 0.70 # Target 70% of parameters for selective QAT (used by score methods)
MIDDLE_EXCLUDE_PERCENT = 0.15 # Exclude ~15% of quantizable layers from start and end

# --- Load Base Model (Keep as is) ---
# Load saved FP32 model
saved_model_dir = 'resnet_fp32_saved_model'
try:
    # Ensure model exists and is runnable
    if not os.path.exists(saved_model_dir):
         raise FileNotFoundError(f"Model directory not found: {saved_model_dir}")
    base_model = tf.keras.models.load_model(saved_model_dir)
    print("Successfully loaded base model.")

    # --- Verify the loaded model can be used ---
    # Try a prediction to catch potential loading/compilation issues
    try:
        dummy_input = tf.random.uniform(shape=(1, 32, 32, 3))
        _ = base_model(dummy_input)
        print("Base model seems runnable.")
    except Exception as run_e:
        print(f"Warning: Loaded model is not immediately runnable: {run_e}")
        print("Attempting to compile the model.")
        # Compile with a basic optimizer/loss/metrics if not already compiled
        try:
            base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            _ = base_model(dummy_input)
            print("Model successfully compiled and runnable.")
        except Exception as compile_run_e:
            print(f"Error: Model is not runnable even after compilation: {compile_run_e}")
            print("This model might have issues. Script may fail later.")


except FileNotFoundError:
    print(f"Error: Saved model directory not found: {saved_model_dir}")
    print("Creating a dummy sequential model for testing purposes.")
    # Create a dummy model with typical layers that can be quantized
    base_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same', name='conv1'),
        tf.keras.layers.BatchNormalization(name='bn1'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', name='conv2'),
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        tf.keras.layers.Dense(10, name='output_dense')
    ])
    # Compile the dummy model
    base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Save the dummy model so it exists next time
    try:
        base_model.save(saved_model_dir)
        print(f"Saved dummy model to {saved_model_dir}.")
    except Exception as save_e:
        print(f"Error saving dummy model: {save_e}")

except Exception as e:
    print(f"Error loading or verifying model: {e}")
    print("Please ensure 'resnet_fp32_saved_model' exists and is a valid Keras SavedModel.")
    print("Exiting.")
    exit() # Ensure script exits if real model loading failed


# --- Prepare Data (Modified for QAT fine-tuning and PTQ calibration) ---
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# --- NEW: Shuffle the training data ---
print("Shuffling training data...")
np.random.seed(42) # Optional: Set a seed for reproducibility
shuffle_indices = np.random.permutation(len(x_train))
x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]
print("Training data shuffled.")
# --- End NEW ---


x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Use a subset of training data for QAT fine-tuning
# Ensure FINETUNE_SIZE + CALIB_SIZE does not exceed x_train length
if FINETUNE_SIZE + CALIB_SIZE > len(x_train):
    print(f"Warning: FINETUNE_SIZE ({FINETUNE_SIZE}) + CALIB_SIZE ({CALIB_SIZE}) exceeds total training data size ({len(x_train)}). Adjusting sizes.")
    CALIB_SIZE = min(CALIB_SIZE, len(x_train) - FINETUNE_SIZE) # Ensure calib_size fits after finetune
    if CALIB_SIZE < 0:
        FINETUNE_SIZE = len(x_train) # Use all data for finetuning if calib_size too big
        CALIB_SIZE = 0
    if FINETUNE_SIZE + CALIB_SIZE > len(x_train): # Re-check if sum still too big
         FINETUNE_SIZE = len(x_train) - CALIB_SIZE # Final adjustment


finetune_data = x_train[:FINETUNE_SIZE]
finetune_labels = y_train[:FINETUNE_SIZE].flatten()

# Use a separate subset for TFLite post-QAT/PTQ conversion calibration
calib_data = x_train[FINETUNE_SIZE : FINETUNE_SIZE + CALIB_SIZE]
calib_labels = y_train[FINETUNE_SIZE : FINETUNE_SIZE + CALIB_SIZE].flatten()

test_images = x_test[:TEST_SIZE]
test_labels = y_test[:TEST_SIZE].flatten()

print(f"Using {len(finetune_data)} images for QAT fine-tuning.")
print(f"Using {len(calib_data)} images for TFLite calibration.")
print(f"Using {len(test_images)} test images for evaluation.")


def representative_dataset():
    """Representative dataset generator for TFLite conversion (PTQ and post-QAT)."""
    # Used by the TFLiteConverter to estimate activation ranges if needed.
    # For QAT, ranges are mostly learned, but providing still helps in conversion.
    # Now uses a larger calibration set.
    for i in range(len(calib_data)):
        yield [calib_data[i:i+1]]


# --- Utility: convert_to_tflite (Simplified for Full INT8 conversion) ---
def convert_to_tflite(saved_model_path, output_path):
    """Converts a Keras SavedModel (PTQ or QAT) to Full INT8 TFLite."""
    if not os.path.isdir(saved_model_path):
        print(f"Error: Saved model directory not found: {saved_model_path}")
        return None # Return None on failure

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

        # Target Full Integer Quantization (INT8 weights and activations)
        # DEFAULT optimization includes quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Explicitly state support for INT8 built-in ops
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # Provide representative dataset for calibration (needed for PTQ activation, good practice for QAT)
        # Now uses the larger calibration set (CALIB_SIZE).
        converter.representative_dataset = representative_dataset

        # Set input/output types to UINT8, which is typical for full INT8 image models [0-255]
        # The converter handles scaling based on calibration/QAT
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8


        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        file_size_mb = os.path.getsize(output_path)/(1024*1024) if os.path.exists(output_path) else 0
        print(f"Saved TFLite model: {output_path} ({file_size_mb:.2f} MB)")
        return output_path # Return path on success
    except Exception as e:
        print(f"Error during TFLite conversion for {output_path}: {e}")
        import traceback
        traceback.print_exc()
        return None # Return None on failure


# --- Helper: Fine-tune QAT model and Save Keras model ---
# Modified to *only* fine-tune and save the Keras model, and return its test accuracy and path
def finetune_and_save_qat_keras_model(qat_prepared_model, finetune_data, finetune_labels, test_data_eval, test_labels_eval, output_prefix):
    """Fine-tunes a QAT-prepared model, evaluates Keras model, and saves it."""
    print(f"\n--- Starting QAT Fine-tuning for {output_prefix} ---")

    # QAT model must be compiled AFTER tfmot.quantization.keras.quantize_apply (or quantize_model)
    # Use the specified QAT learning rate
    qat_prepared_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=QAT_LEARNING_RATE), # Use Adam with reduced LR
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), # Use Sparse Categorical Crossentropy
        metrics=['accuracy']
    )
    print("QAT model compiled with Adam optimizer and learning rate", QAT_LEARNING_RATE)

    # Unfreeze BN layers for QAT fine-tuning
    # Note: In QAT, BN layers are replaced with QuantizeWrapperV2 containing BatchNormalization.
    # We need to find the inner BN layer or unfreeze the wrapper if it handles training.
    # Let's try unfreezing the wrapper itself and the inner layer if accessible.
    print("Unfreezing BatchNormalization layers (and wrappers)...")
    for layer in qat_prepared_model.layers:
        # Check for QuantizeWrapperV2 containing BN
        if isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2):
             if isinstance(layer.layer, tf.keras.layers.BatchNormalization):
                 layer.trainable = True # Unfreeze wrapper
                 layer.layer.trainable = True # Unfreeze inner BN
                 #print(f" - Unfrozen BN wrapper+layer: {layer.name}")
             else:
                 # Unfreeze other wrappers as well, as they contain trainable quantizer parameters
                 layer.trainable = True
                 #print(f" - Unfrozen Wrapper: {layer.name}")
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
             layer.trainable = True # Unfreeze standalone BN
             #print(f" - Unfrozen standalone BN layer: {layer.name}")
        # Add other trainable layers that are not quantized if needed, but QAT often traines only quantized parts
        # For this script's purpose, we focus on BN which is critical for QAT convergence
        # Original base_model.trainable_variables controls what was trainable initially


    print(f"Fine-tuning for {QAT_EPOCHS} epochs...")
    try:
        history = qat_prepared_model.fit(
            finetune_data,
            finetune_labels,
            epochs=QAT_EPOCHS,
            batch_size=32, # Appropriate batch size
            verbose=1, # Show training progress
            validation_data=(test_data_eval, test_labels_eval) # Added validation data
        )
        print("Fine-tuning complete.")
        # You might want to print/log history.history['val_accuracy'][-1] etc.
    except Exception as e:
        print(f"Error during QAT fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return None, None # Return None for both path and Keras accuracy


    # --- Evaluate QAT Keras Model (with Fake Quant) before Conversion ---
    print(f"\n--- Evaluating {output_prefix} QAT Keras Model (with Fake Quant) ---")
    try:
        # Use the same evaluation data as the final TFLite evaluation
        keras_loss, keras_accuracy = qat_prepared_model.evaluate(test_data_eval, test_labels_eval, verbose=0)
        print(f"{output_prefix} QAT Keras Model Accuracy: {keras_accuracy*100:.2f}% ({int(keras_accuracy*len(test_labels_eval))}/{len(test_labels_eval)})")
    except Exception as e:
         print(f"Error evaluating {output_prefix} QAT Keras model: {e}")
         import traceback
         traceback.print_exc()
         keras_accuracy = None # Mark accuracy as failed


    # Save the fine-tuned QAT Keras model
    saved_qat_model_dir = f'{output_prefix}_qat_saved_model'
    # Clean up previous saved model directory if it exists
    if os.path.exists(saved_qat_model_dir):
         try:
             shutil.rmtree(saved_qat_model_dir)
             # print(f"Cleaned up previous directory: {saved_qat_model_dir}")
         except Exception as clean_e:
             print(f"Warning: Could not clean up directory {saved_qat_model_dir}: {clean_e}")

    try:
        qat_prepared_model.save(saved_qat_model_dir)
        print(f"Saved fine-tuned QAT Keras model: {saved_qat_model_dir}")
        return saved_qat_model_dir, keras_accuracy # Return path to saved model and Keras accuracy
    except Exception as e:
        print(f"Error saving fine-tuned QAT model: {e}")
        # Clean up potentially partially created directory
        if os.path.exists(saved_qat_model_dir):
             shutil.rmtree(saved_qat_model_dir)
        return None, keras_accuracy # Return None for path, but return Keras accuracy if available


# --- Helper Function for Percentage-Based Selection ---
def select_layers_by_percentage(model, score_dict, target_percentage=0.7, quantize_lowest=True):
    """
    Selects layers for quantization based on a score until a target percentage
    of parameters is reached. Used by score-based methods (L2, Hessian, Hybrid).
    """
    quantizable_layers_with_scores = []
    total_quantizable_params_with_scores = 0

    # Identify layers that are typically quantizable and have parameters AND a score
    quantizable_types = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)

    for layer in model.layers:
        if isinstance(layer, quantizable_types):
             params = layer.count_params()
             # Only consider layers with parameters and ensure they have a score
             if params > 0 and layer.name in score_dict:
                  quantizable_layers_with_scores.append({
                       'name': layer.name,
                       'score': score_dict[layer.name],
                       'params': params
                   })
                  total_quantizable_params_with_scores += params
             # else: layers without params or score are not candidates for this selection method


    if total_quantizable_params_with_scores == 0:
        print("No quantizable parameters with scores found for selection by percentage.")
        return set(), 0.0

    # Sort layers based on score and the quantize_lowest flag
    sorted_layers = sorted(quantizable_layers_with_scores, key=lambda x: x['score'], reverse=not quantize_lowest)

    target_params = total_quantizable_params_with_scores * target_percentage
    selected_layers_set = set()
    accumulated_params = 0

    print(f"\nTargeting {target_percentage*100:.1f}% ({int(target_params)}) of {total_quantizable_params_with_scores} total quantizable parameters with scores.")
    print(f"Sorting layers by score ({'Ascending - quantizing lowest scores first' if quantize_lowest else 'Descending - quantizing highest scores first'})...")

    for layer_info in sorted_layers:
        if accumulated_params < target_params:
            selected_layers_set.add(layer_info['name'])
            accumulated_params += layer_info['params']
            # print(f"  Selecting {layer_info['name']} (Score: {layer_info['score']:.4f}, Params: {layer_info['params']}) -> Accumulated params: {accumulated_params}")
        else:
             # Stop once target is reached or exceeded, but the last layer added might exceed the target
             break

    actual_percentage = (accumulated_params / total_quantizable_params_with_scores) if total_quantizable_params_with_scores > 0 else 0
    print(f"Selected {len(selected_layers_set)} layers for QAT based on percentage target.")
    print(f"Covered {accumulated_params} parameters ({actual_percentage*100:.2f}% of total quantizable parameters with scores).")

    return selected_layers_set, actual_percentage

# --- Helper to get the annotated model for selective QAT ---
def get_annotated_model_selective(model, selected_layer_names):
     QuantAnnotate = tfmot.quantization.keras.quantize_annotate_layer
     def apply_quantization_based_on_selection(layer):
         # Check if the layer is quantizable and in the selected list
         quantizable_types = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)
         # Also include BatchNormalization layers within QuantizeWrapperV2 if their preceding layer is selected
         # Or more simply, annotate all quantizable layers (Conv, Dense, etc.) and let TFMOT handle BN
         # A common pattern is to annotate Conv/Dense and TFMOT automatically handles BN folding/wrapping
         # Let's stick to annotating the main quantizable types (Conv, Dense)
         if isinstance(layer, quantizable_types) and layer.name in selected_layer_names:
             # print(f"  -> Annotating {layer.name} for QAT (Selected)")
             return QuantAnnotate(layer)
         # Return non-quantizable layers or non-selected quantizable layers as is
         return layer

     # Clone and annotate the model
     print(f"Creating annotated model...")
     annotated_model = tf.keras.models.clone_model(
         model, clone_function=apply_quantization_based_on_selection
     )
     print("Annotation complete.")
     return annotated_model


# --- Prepare Full QAT Model ---
def prepare_qat_full(model):
    """Applies QAT preparation to the entire model using quantize_model."""
    print(f"\n--- Starting Full QAT Preparation ---")

    # Use quantize_model to automatically annotate and apply quantization to all supported layers
    with tfmot.quantization.keras.quantize_scope():
        # quantize_model handles annotation and applying quantization
        qat_model = tfmot.quantization.keras.quantize_model(model)

    print("Full QAT model prepared.")
    # For Full QAT, the percentage is effectively 100% of quantizable layers with params
    total_quantizable_params = 0
    quantizable_types = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)
    for layer in model.layers:
         if isinstance(layer, quantizable_types):
              total_quantizable_params += layer.count_params() if layer.count_params() > 0 else 0

    # Return the model and 1.0 for 100% coverage of quantizable params with scores (all of them here)
    return qat_model, 1.0


# --- Scoring Functions (Used by Selective Methods) ---
# Define common quantizable types for score calculation and selection consistency
QUANTIZABLE_SCORE_TYPES = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)


# 1) Weight L2-norm based selective QAT (Percentage)
def prepare_qat_l2_percent(model, target_percentage=0.7):
    print(f"\n--- Starting L2 Norm Percentage Selective QAT Preparation ({target_percentage*100:.0f}%) ---")
    layer_scores = {}

    for layer in model.layers:
        if isinstance(layer, QUANTIZABLE_SCORE_TYPES):
            weights = layer.get_weights()
            if weights and len(weights) > 0 and weights[0] is not None:
                try:
                    l2_norm = np.linalg.norm(weights[0].flatten())
                    layer_scores[layer.name] = l2_norm
                except Exception as e:
                    print(f"Could not process layer {layer.name} for L2 norm: {e}")
            # Layers without weights[0] are implicitly not included in layer_scores


    selected_layers, actual_percentage = select_layers_by_percentage(
        model, layer_scores, target_percentage, quantize_lowest=True # Quantize low L2 norm layers
    )

    if not selected_layers:
        print("No layers selected for L2 norm QAT. Aborting preparation.")
        return None, None # Return None for model and percentage

    annotated_model = get_annotated_model_selective(model, selected_layers)

    # Apply quantization to get the QAT-prepared model
    with tfmot.quantization.keras.quantize_scope():
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    print("QAT model prepared (L2 Norm Percentage).")
    return qat_model, actual_percentage # Return actual percentage


# 3) Hessian-based sensitivity selective QAT (Percentage)
def prepare_qat_hessian_percent(model, target_percentage=0.7):
    print(f"\n--- Starting Hessian (Gradient Approx) Percentage Selective QAT Preparation ({target_percentage*100:.0f}%) ---")
    layer_sens = {}
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    # Use a small batch from the fine-tuning data for gradient calculation
    sample_images = finetune_data[:100] # Use a subset, smaller than full finetune_data
    sample_labels = finetune_labels[:100]
    param_to_layer = {} # Map variable references to layer names

    # Identify quantizable layers with trainable variables that might contribute to scores
    quantizable_layers_with_trainable_params = [
        layer for layer in model.layers if isinstance(layer, QUANTIZABLE_SCORE_TYPES) and layer.trainable_variables
    ]

    if not quantizable_layers_with_trainable_params:
         print("No quantizable layers with trainable parameters found to calculate Hessian scores.")
         return None, None

    for layer in quantizable_layers_with_trainable_params:
        for param in layer.trainable_variables: # Get trainable variables *for this layer*
             param_to_layer[param.ref()] = layer.name # Use variable reference as key


    try:
        print(f"Calculating gradients with input shape: {sample_images.shape}")
        # Ensure the model is built before calculating gradients
        _ = model(tf.random.uniform(shape=(1,) + model.input_shape[1:])) # Build the model if not already built

        with tf.GradientTape() as tape:
            # Watch all trainable variables explicitly to be safe (optional but can help)
            # tape.watch(model.trainable_variables)
            preds = model(sample_images, training=False) # Use training=False for inference path gradients
            loss = loss_fn(sample_labels, preds)
        print(f"Calculated loss: {loss.numpy()}")

        # Get gradients for trainable variables associated with the relevant layers
        trainable_vars = [v for v in model.trainable_variables if v.ref() in param_to_layer]
        grads = tape.gradient(loss, trainable_vars)

        if grads is None or not any(g is not None for g in grads):
            print("Error: Gradients are None. Cannot calculate sensitivity. Check model, loss, data.")
            return None, None

        # Aggregate scores by module name
        for var, grad in zip(trainable_vars, grads):
            if grad is None: continue # Skip if gradient is None

            layer_name = param_to_layer[var.ref()] # Get layer name from map
            # Ensure the variable belongs to a quantizable layer type we care about and has a mapping
            if layer_name is not None and layer_name in {l.name for l in model.layers if isinstance(l, QUANTIZABLE_SCORE_TYPES)}:
                 # Approximation: Mean of (Grad * Parameter)^2
                 # Ensure grad and var have compatible shapes or broadcast
                 # If grad/var are tensors, tf.square(grad * var) works fine
                 score = tf.reduce_mean(tf.square(grad * var)).numpy()
                 # Aggregate score by layer name (a layer might have weight and bias)
                 layer_sens[layer_name] = layer_sens.get(layer_name, 0) + score


        # Ensure all potential quantizable layers with trainable params have a score entry, even if grad was 0
        all_potential_quantizable_names = {l.name for l in quantizable_layers_with_trainable_params}
        for name in all_potential_quantizable_names:
            if name not in layer_sens:
                 layer_sens[name] = 0 # Assign 0 if no variables or grad found for this layer


        if not layer_sens:
            print("Error: Sensitivity scores dictionary is empty after calculation.")
            return None, None

        selected_layers, actual_percentage = select_layers_by_percentage(
            model, layer_sens, target_percentage, quantize_lowest=True # Quantize low sensitivity layers
        )

        if not selected_layers:
            print("No layers selected for Hessian QAT based on scores/percentage. Aborting preparation.")
            return None, None

        annotated_model = get_annotated_model_selective(model, selected_layers)

        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

        print("QAT model prepared (Hessian Percentage).")
        return qat_model, actual_percentage


    except Exception as e:
        print(f"Error during Hessian percentage QAT preparation: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# 4) Hybrid selective QAT (Percentage)
# Activation Range calculation using an intermediate model
def calculate_activation_ranges(model, sample_data):
    print("Calculating Activation Ranges...")
    act_range = {}
    quantizable_layer_names_with_output = set() # Track quantizable layers whose output we *can* capture

    # Identify quantizable layers and try to get their output tensors
    layer_outputs_dict = {}
    quantizable_types = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)

    # Ensure the model is built before trying to get intermediate outputs
    _ = model(tf.random.uniform(shape=(1,) + model.input_shape[1:])) # Build the model if not already built


    for layer in model.layers:
        if isinstance(layer, quantizable_types):
             # Attempt to get output tensor - might fail for some complex layer types or graph structures
             try:
                 layer_outputs_dict[layer.name] = layer.output
                 quantizable_layer_names_with_output.add(layer.name) # Only add if output is accessible
             except AttributeError:
                 # print(f"Warning: Layer {layer.name} does not have a direct '.output' attribute needed for activation range calculation.")
                 pass # Skip layers where output is not easily accessible


    if not layer_outputs_dict:
         print("Warning: No quantizable layers with accessible outputs found for activation range calculation.")
         # All ranges will be 0 after the loop below


    # Create a functional API model for intermediate outputs if outputs were captured
    intermediate_model = None
    if layer_outputs_dict:
        try:
             # Ensure the model is buildable with the desired outputs
             # Use the original model's inputs
             intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs_dict)
             print(f"Calculating activations with input shape: {sample_data.shape}")
             activations_dict = intermediate_model.predict(sample_data, verbose=0)

             for layer_name, act in activations_dict.items():
                 # Ensure the captured activation corresponds to a quantizable layer we intended
                 if layer_name in quantizable_layer_names_with_output and isinstance(act, np.ndarray) and np.issubdtype(act.dtype, np.number):
                     min_val = np.min(act)
                     max_val = np.max(act)
                     act_range[layer_name] = max_val - min_val
                 else:
                     # Should not happen if keys match, but safety check
                     act_range[layer_name] = 0 # Assign 0 if not numeric or problematic

        except Exception as e:
            print(f"Error during intermediate model prediction for activation ranges: {e}")
            import traceback
            traceback.print_exc()
            # Assign 0 range to all if prediction fails

    # Ensure all quantizable layers (found initially) have a range entry, even if 0
    # Use QUANTIZABLE_SCORE_TYPES to get all potential layers for scoring
    all_potential_quantizable_names = {layer.name for layer in model.layers if isinstance(layer, QUANTIZABLE_SCORE_TYPES)}
    for name in all_potential_quantizable_names:
        if name not in act_range:
            act_range[name] = 0 # Default to 0 if range calculation failed for a layer or no outputs captured initially


    print("Activation range calculation finished.")
    # print("Activation ranges:", act_range)
    return act_range


def prepare_qat_hybrid_percent(model, alpha=0.4, beta=0.4, gamma=0.2, target_percentage=0.7):
    print(f"\n--- Starting Hybrid Percentage Selective QAT Preparation ({target_percentage*100:.0f}%) ---")

    # Identify all potential quantizable layers for scoring
    all_potential_quantizable_names = {layer.name for layer in model.layers if isinstance(layer, QUANTIZABLE_SCORE_TYPES)}
    param_to_layer = {} # Map variable references to layer names for sensitivity
    quantizable_layers_with_trainable_params = [
        layer for layer in model.layers if isinstance(layer, QUANTIZABLE_SCORE_TYPES) and layer.trainable_variables
    ]

    for layer in quantizable_layers_with_trainable_params:
         for param in layer.trainable_variables:
              param_to_layer[param.ref()] = layer.name


    if not all_potential_quantizable_names:
         print("No quantizable layers found to calculate Hybrid scores.")
         return None, None


    # Use a small batch from fine-tuning data for score calculations
    sample_data_scores = finetune_data[:100]
    sample_labels_scores = finetune_labels[:100]


    # --- Calculate L2 norms ---
    l2_norms = {}
    for layer in model.layers: # Iterate through model layers
        if layer.name in all_potential_quantizable_names: # Only consider our target types
            weights = layer.get_weights()
            if weights and len(weights) > 0 and weights[0] is not None:
                try:
                    l2_norm = np.linalg.norm(weights[0].flatten())
                    layer_scores[layer.name] = l2_norm
                except Exception as e:
                    print(f"Could not process layer {layer.name} for L2 norm: {e}")
            else:
                 l2_norms[layer.name] = 0 # Assign 0 if no weights[0]


    # Ensure all potential quantizable layers have an L2 entry, even if 0
    for name in all_potential_quantizable_names:
         if name not in l2_norms:
              l2_norms[name] = 0


    max_l2 = max(l2_norms.values()) if l2_norms else 1.0
    # print(f"Calculated L2 norms for {len(l2_norms)} layers. Max L2: {max_l2:.4f}")


    # --- Calculate activation ranges ---
    # Pass the small sample data batch
    act_ranges = calculate_activation_ranges(model, sample_data_scores)
    # Ensure all potential quantizable layers have an entry even if range calc failed
    for name in all_potential_quantizable_names:
        if name not in act_ranges:
            act_ranges[name] = 0 # Should be covered by calculate_activation_ranges, but safety check

    max_range = max(act_ranges.values()) if act_ranges else 1.0
    # print(f"Calculated activation ranges for {len(act_ranges)} layers. Max Range: {max_range:.4f}")


    # --- Calculate sensitivity (reuse gradient approach) ---
    layer_sens = {}
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    try:
        print(f"Calculating gradients with input shape: {sample_data_scores.shape}")
         # Ensure the model is built before calculating gradients
        _ = model(tf.random.uniform(shape=(1,) + model.input_shape[1:])) # Build the model if not already built

        with tf.GradientTape() as tape:
             preds = model(sample_data_scores, training=False)
             loss = loss_fn(sample_labels_scores, preds)

        # Get gradients for trainable variables associated with the relevant layers
        trainable_vars = [v for v in model.trainable_variables if v.ref() in param_to_layer]
        grads = tape.gradient(loss, trainable_vars)

        if grads is None or not any(g is not None for g in grads):
            print("Warning: Gradients are None for sensitivity calculation in hybrid.")
        else:
            # Aggregate scores by module name using the pre-built map
            for var, grad in zip(trainable_vars, grads):
                if grad is None: continue
                layer_name = param_to_layer[var.ref()] # Use the map
                # Ensure the variable belongs to a quantizable layer type we care about
                if layer_name is not None and layer_name in all_potential_quantizable_names:
                    score = tf.reduce_mean(tf.square(grad * var)).numpy()
                    layer_sens[layer_name] = layer_sens.get(layer_name, 0) + score

        # Ensure all potential quantizable layers have a sensitivity entry even if grad was None or map failed
        for name in all_potential_quantizable_names:
             if name not in layer_sens:
                  layer_sens[name] = 0


        max_sens = max(layer_sens.values()) if layer_sens else 1.0
        # print(f"Calculated sensitivity for relevant layers. Max Sensitivity: {max_sens:.6f}")

    except Exception as e:
        print(f"Error during sensitivity calculation for hybrid: {e}")
        # Assign 0 sensitivity if calculation fails
        for name in all_potential_quantizable_names:
            layer_sens[name] = 0
        max_sens = 1.0


    # --- Compute hybrid score ---
    hybrid_scores = {}
    print("\nCalculating Hybrid Scores:")
    # Only calculate hybrid score for identified quantizable layers
    for layer_name in all_potential_quantizable_names:
        # Normalize scores (handle division by zero)
        norm_l2 = (l2_norms.get(layer_name, 0) / max_l2) if max_l2 != 0 else 0
        norm_range = (act_ranges.get(layer_name, 0) / max_range) if max_range != 0 else 0
        norm_sens = (layer_sens.get(layer_name, 0) / max_sens) if max_sens != 0 else 0

        # Combine scores - assuming lower is better for quantization
        score = alpha * norm_l2 + beta * norm_range + gamma * norm_sens
        hybrid_scores[layer_name] = score
        # print(f"  Layer {layer_name}: NormL2={norm_l2:.3f}, NormRange={norm_range:.3f}, NormSens={norm_sens:.3f} -> Hybrid Score = {score:.4f}")

    # Select layers based on the lowest hybrid scores
    selected_layers, actual_percentage = select_layers_by_percentage(
        model, hybrid_scores, target_percentage, quantize_lowest=True # Quantize low score layers
    )

    if not selected_layers:
        print("No layers selected for Hybrid QAT based on scores/percentage. Aborting preparation.")
        return None, None

    annotated_model = get_annotated_model_selective(model, selected_layers)

    with tfmot.quantization.keras.quantize_scope():
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    print("QAT model prepared (Hybrid Percentage).")
    return qat_model, actual_percentage


# --- New Selective QAT Method: Quantize Middle Layers by Index ---
def prepare_qat_middle_percent(model, exclude_percent=0.15):
    """
    Applies QAT to layers in the middle range, excluding a percentage
    of quantizable layers from the start and end based on index.
    """
    print(f"\n--- Starting Middle Layers Selective QAT Preparation (Excluding ~{exclude_percent*100:.0f}% Start/End) ---")

    quantizable_layers_ordered = []
    total_quantizable_params = 0
    quantizable_types = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)

    # Identify quantizable layers in the model's sequential order that have parameters
    for layer in model.layers:
         if isinstance(layer, quantizable_types) and layer.count_params() > 0:
              quantizable_layers_ordered.append(layer)
              total_quantizable_params += layer.count_params()


    num_quantizable_layers = len(quantizable_layers_ordered)
    if num_quantizable_layers == 0:
        print("No quantizable layers found with parameters for middle layer selection.")
        return None, 0.0

    num_exclude_start = int(num_quantizable_layers * exclude_percent)
    num_exclude_end = int(num_quantizable_layers * exclude_percent)

    # Ensure we don't exclude more layers than exist or exclude the entire model
    # If exclusion is too high, quantize all layers instead of none/very few unexpectedly
    if num_exclude_start + num_exclude_end >= num_quantizable_layers:
        print(f"Warning: Exclusion percentages ({exclude_percent*100}% start + {exclude_percent*100}% end) are too high ({num_exclude_start + num_exclude_end} total layers to exclude) for {num_quantizable_layers} quantizable layers.")
        print("Quantizing ALL quantizable layers instead.")
        selected_layers_list = quantizable_layers_ordered # Select all
        num_exclude_start = 0 # Reset exclusion counts for reporting
        num_exclude_end = 0
    else:
         # Select layers in the middle by index
         selected_layers_list = quantizable_layers_ordered[num_exclude_start : num_quantizable_layers - num_exclude_end]


    selected_layer_names = {layer.name for layer in selected_layers_list}

    if not selected_layer_names:
         print("No layers selected for QAT after excluding start/end layers based on index. Aborting preparation.")
         return None, 0.0

    # Calculate actual parameter percentage covered by the selected layers
    selected_params = sum(layer.count_params() for layer in selected_layers_list)
    actual_percentage = (selected_params / total_quantizable_params) if total_quantizable_params > 0 else 0

    print(f"Selected {len(selected_layers_list)} layers for QAT (indices {num_exclude_start} to {num_quantizable_layers - num_exclude_end - 1 if num_quantizable_layers > 0 else -1}).")
    print(f"Covered {selected_params} parameters ({actual_percentage*100:.2f}% of total quantizable parameters with scores).")


    # Annotate the model based on the selected layer names
    annotated_model = get_annotated_model_selective(model, selected_layer_names)

    # Apply quantization to get the QAT-prepared model
    with tfmot.quantization.keras.quantize_scope():
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    print("QAT model prepared (Middle Layers by Index).")
    return qat_model, actual_percentage # Return QAT model and actual parameter percentage


# --- Full INT8 Post-Training Quantization (Baseline - Keep as is) ---
def quantize_activation_ptq(model):
    print("\n--- Starting Full INT8 Post-Training Quantization (PTQ Baseline) ---")
    save_path = 'model_activation_int8_ptq_keras_temp'
    tflite_path_out = 'resnet_activation_int8_ptq.tflite' # Use a different variable name here
    try:
        # Save the original FP32 model temporarily
        # Clean up previous saved model directory if it exists
        if os.path.exists(save_path):
             try:
                 shutil.rmtree(save_path)
                 # print(f"Cleaned up previous PTQ directory: {save_path}")
             except Exception as clean_e:
                 print(f"Warning: Could not clean up PTQ directory {save_path}: {clean_e}")

        model.save(save_path)
        print(f"Saved temporary Keras model for PTQ: {save_path}")
        # Convert using convert_to_tflite
        # Pass the desired output path directly to convert_to_tflite
        converted_tflite_path = convert_to_tflite(save_path, tflite_path_out)

        # Clean up temporary Keras model directory after conversion (optional but good practice)
        if os.path.exists(save_path):
             try:
                 shutil.rmtree(save_path)
                 # print(f"Cleaned up temporary directory: {save_path}")
             except Exception as clean_e:
                 print(f"Error cleaning up temporary directory {save_path}: {clean_e}")


        return converted_tflite_path # Return the path generated by convert_to_tflite
    except Exception as e:
        print(f"Error saving or converting model for PTQ: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Evaluate TFLite model (Keep as is) ---
def evaluate_tflite(tflite_path, test_images, test_labels):
    """Evaluates a TFLite model and returns accuracy."""
    if not os.path.exists(tflite_path):
        print(f"Skipping evaluation: TFLite file not found: {tflite_path}")
        return None # Return None if file not found

    print(f"\n--- Evaluating {os.path.basename(tflite_path)} ---")
    try:
        # Ensure test_images are the expected float type for the converter
        # The converter handles the final input type (e.g., uint8 for INT8)
        test_images_eval = test_images.astype(np.float32) # Should already be float32 [0,1] from data prep

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        input_is_int = np.issubdtype(input_details['dtype'], np.integer)
        # print(f"Model Input Type: {input_details['dtype']}, Expects Int: {input_is_int}")
        # print(f"Model Output Type: {output_details['dtype']}")


        correct = 0
        num_eval_images = len(test_images_eval)
        # printed_warning = False # Warning about (0,0) params removed as it's less relevant for standard INT8


        for i in range(num_eval_images):
            img = test_images_eval[i] # Should be float [0, 1]
            label = test_labels[i]

            # Prepare input tensor based on interpreter's expected type and shape
            input_shape = input_details['shape']
            img_batch = np.expand_dims(img, axis=0) # Add batch dimension


            if input_is_int:
                # Quantize the input image to the expected integer type
                scale, zero_point = input_details['quantization']
                # Standard per-tensor quantization formula: int = float / scale + zero_point
                # Ensure casting to the interpreter's required dtype
                # Add a small epsilon or check scale to avoid division by zero if scale is unexpectedly 0
                scale = scale if scale != 0 else 1e-8
                inp = (img_batch / scale + zero_point).astype(input_details['dtype'])

            else:
                # Input is float, just ensure correct dtype
                 inp = img_batch.astype(input_details['dtype'])


            # Ensure input shape matches model's expected input shape
            if inp.shape != tuple(input_shape):
                 print(f"\nError: Input shape mismatch for image {i}. Expected {input_shape}, got {inp.shape}")
                 # Attempt a reshape - risky
                 # inp = inp.reshape(input_shape)
                 print(f"Skipping image {i} due to shape mismatch.")
                 continue # Skip this image

            interpreter.set_tensor(input_details['index'], inp)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details['index'])

            # Dequantize the output if necessary
            output_is_int = np.issubdtype(output_details['dtype'], np.integer)
            if output_is_int:
                scale, zero_point = output_details['quantization']
                # Standard per-tensor dequantization formula: float = (int - zero_point) * scale
                scale = scale if scale != 0 else 1e-8 # Avoid division by zero
                dequantized_output = (output_data.astype(np.float32) - zero_point) * scale
                pred = np.argmax(dequantized_output[0])
            else:
                # Float output
                pred = np.argmax(output_data[0])

            if pred == label:
                correct += 1

            print(f"Evaluating: {i+1}/{num_eval_images}", end='\r')

        accuracy = correct / num_eval_images if num_eval_images > 0 else 0
        # Clear the progress line before printing final accuracy
        print(" " * 50, end='\r')
        print(f"{os.path.basename(tflite_path)} -> Accuracy: {accuracy*100:.2f}% ({correct}/{num_eval_images})")
        return accuracy # Return accuracy

    except Exception as e:
        print(f"\nError evaluating {tflite_path}: {e}")
        import traceback
        traceback.print_exc()
        return None # Return None if evaluation fails


# --- Evaluate FP32 Keras Model ---
def evaluate_keras_model(model, test_images, test_labels):
    """Evaluates a Keras model and returns accuracy."""
    print("\n--- Evaluating FP32 Keras Base Model ---")
    try:
        # Ensure the model is compiled for evaluation
        # Use same compile settings as used for QAT/PTQ models if possible
        # Check if model is already compiled before compiling again
        if not hasattr(model, 'optimizer'):
             print("Compiling FP32 model for evaluation...")
             # Use the same optimizer setup as QAT for consistency if possible, but standard Adam is usually fine
             model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categoricalCrossentropy', metrics=['accuracy'])


        loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print(f"FP32 Keras Base Model Accuracy: {accuracy*100:.2f}% ({int(accuracy*len(test_labels))}/{len(test_labels)})")
        return accuracy
    except Exception as e:
        print(f"Error evaluating Keras model: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Main Execution ---
if __name__ == "__main__":
    if 'base_model' not in locals() and 'base_model' not in globals():
        print("Base model was not loaded successfully. Cannot proceed.")
        exit() # Ensure script exits if model loading failed

    evaluation_results = {} # Store evaluation results by name

    # 1. Evaluate Original FP32 Model
    fp32_accuracy = evaluate_keras_model(base_model, test_images, test_labels)
    if fp32_accuracy is not None:
        evaluation_results['FP32_Base_Model'] = fp32_accuracy

    # 2. Prepare, Fine-tune, Save Keras, Convert, and Evaluate Full QAT Model
    print("\n--- Processing Full QAT ---")
    # Need to clone the base model before applying quantize_model to avoid modifying the original
    base_model_for_full_qat = tf.keras.models.clone_model(base_model)
    base_model_for_full_qat.set_weights(base_model.get_weights()) # Copy weights

    # Prepare QAT model
    qat_prepared_full, actual_percent_full = prepare_qat_full(base_model_for_full_qat) # Pass the cloned model

    if qat_prepared_full is not None:
         # Use actual percentage (should be near 100) in filename
         output_prefix = f'resnet_full_qat_{int(actual_percent_full*100)}percent'
         # Fine-tune and Save Keras model
         saved_keras_qat_path, keras_accuracy_qat_full = finetune_and_save_qat_keras_model( # Capture Keras accuracy
              qat_prepared_full,
              finetune_data, finetune_labels, test_images, test_labels, output_prefix
         )
         if keras_accuracy_qat_full is not None: # Store Keras accuracy
              evaluation_results['Full_INT8_QAT_Keras_FakeQuant'] = keras_accuracy_qat_full

         # Convert Saved Keras model to TFLite
         if saved_keras_qat_path:
              tflite_path = convert_to_tflite(saved_keras_qat_path, f'{output_prefix}_qat.tflite')
              # Evaluate the TFLite model
              if tflite_path:
                   tflite_accuracy = evaluate_tflite(tflite_path, test_images, test_labels)
                   if tflite_accuracy is not None:
                        evaluation_results['Full_INT8_QAT_TFLite'] = tflite_accuracy
              else:
                   print(f"{output_prefix} TFLite conversion failed.")
         else:
             print(f"{output_prefix} Keras model saving/fine-tuning failed.")
    else:
         print("Full QAT preparation failed.")


    # 3. Prepare, Fine-tune, Save Keras, Convert, and Evaluate Selective QAT Methods
    print(f"\nStarting Selective QAT Procedures (Score-based with Target Percentage: {TARGET_QUANT_PERCENTAGE*100:.0f}% & Index-based Excluding {MIDDLE_EXCLUDE_PERCENT*100:.0f}% Start/End)")

    # L2 Norm Selective QAT
    print("\n--- Processing Selective L2 Norm QAT ---")
    # Clone the base model for each selective method to start fresh
    base_model_for_l2_qat = tf.keras.models.clone_model(base_model)
    base_model_for_l2_qat.set_weights(base_model.get_weights()) # Copy weights
    qat_prepared_l2, actual_percent_l2 = prepare_qat_l2_percent(base_model_for_l2_qat, target_percentage=TARGET_QUANT_PERCENTAGE)
    if qat_prepared_l2 is not None:
         output_prefix = f'resnet_l2norm_qat_{int(actual_percent_l2*100)}percent'
         saved_keras_qat_path, keras_accuracy_qat_l2 = finetune_and_save_qat_keras_model( # Capture Keras accuracy
              qat_prepared_l2,
              finetune_data, finetune_labels, test_images, test_labels, output_prefix
         )
         if keras_accuracy_qat_l2 is not None: # Store Keras accuracy
              evaluation_results[f'Selective_L2_QAT_{int(actual_percent_l2*100)}Percent_Keras_FakeQuant'] = keras_accuracy_qat_l2

         # Convert Saved Keras model to TFLite
         if saved_keras_qat_path:
              tflite_path = convert_to_tflite(saved_keras_qat_path, f'{output_prefix}_qat.tflite')
              # Evaluate the TFLite model
              if tflite_path:
                   tflite_accuracy = evaluate_tflite(tflite_path, test_images, test_labels)
                   if tflite_accuracy is not None:
                        evaluation_results[f'Selective_L2_QAT_{int(actual_percent_l2*100)}Percent_TFLite'] = tflite_accuracy
              else:
                   print(f"{output_prefix} TFLite conversion failed.")
         else:
              print(f"{output_prefix} Keras model saving/fine-tuning failed.")
    else:
        print("Selective L2 QAT preparation failed.")

    # Hessian Selective QAT
    print("\n--- Processing Selective Hessian QAT ---")
    base_model_for_hessian_qat = tf.keras.models.clone_model(base_model)
    base_model_for_hessian_qat.set_weights(base_model.get_weights()) # Copy weights
    qat_prepared_hessian, actual_percent_hessian = prepare_qat_hessian_percent(base_model_for_hessian_qat, target_percentage=TARGET_QUANT_PERCENTAGE)
    if qat_prepared_hessian is not None:
         output_prefix = f'resnet_hessian_qat_{int(actual_percent_hessian*100)}percent'
         saved_keras_qat_path, keras_accuracy_qat_hessian = finetune_and_save_qat_keras_model( # Capture Keras accuracy
              qat_prepared_hessian,
              finetune_data, finetune_labels, test_images, test_labels, output_prefix
         )
         if keras_accuracy_qat_hessian is not None: # Store Keras accuracy
              evaluation_results[f'Selective_Hessian_QAT_{int(actual_percent_hessian*100)}Percent_Keras_FakeQuant'] = keras_accuracy_qat_hessian

         # Convert Saved Keras model to TFLite
         if saved_keras_qat_path:
              tflite_path = convert_to_tflite(saved_keras_qat_path, f'{output_prefix}_qat.tflite')
              # Evaluate the TFLite model
              if tflite_path:
                   tflite_accuracy = evaluate_tflite(tflite_path, test_images, test_labels)
                   if tflite_accuracy is not None:
                        evaluation_results[f'Selective_Hessian_QAT_{int(actual_percent_hessian*100)}Percent_TFLite'] = tflite_accuracy
              else:
                   print(f"{output_prefix} TFLite conversion failed.")
         else:
              print(f"{output_prefix} Keras model saving/fine-tuning failed.")
    else:
        print("Selective Hessian QAT preparation failed.")

    # Hybrid Selective QAT
    print("\n--- Processing Selective Hybrid QAT ---")
    base_model_for_hybrid_qat = tf.keras.models.clone_model(base_model)
    base_model_for_hybrid_qat.set_weights(base_model.get_weights()) # Copy weights
    qat_prepared_hybrid, actual_percent_hybrid = prepare_qat_hybrid_percent(base_model_for_hybrid_qat, target_percentage=TARGET_QUANT_PERCENTAGE)
    if qat_prepared_hybrid is not None:
         output_prefix = f'resnet_hybrid_qat_{int(actual_percent_hybrid*100)}percent'
         saved_keras_qat_path, keras_accuracy_qat_hybrid = finetune_and_save_qat_keras_model( # Capture Keras accuracy
              qat_prepared_hybrid,
              finetune_data, finetune_labels, test_images, test_labels, output_prefix
         )
         if keras_accuracy_qat_hybrid is not None: # Store Keras accuracy
              evaluation_results[f'Selective_Hybrid_QAT_{int(actual_percent_hybrid*100)}Percent_Keras_FakeQuant'] = keras_accuracy_qat_hybrid

         # Convert Saved Keras model to TFLite
         if saved_keras_qat_path:
              tflite_path = convert_to_tflite(saved_keras_qat_path, f'{output_prefix}_qat.tflite')
              # Evaluate the TFLite model
              if tflite_path:
                   tflite_accuracy = evaluate_tflite(tflite_path, test_images, test_labels)
                   if tflite_accuracy is not None:
                        evaluation_results[f'Selective_Hybrid_QAT_{int(actual_percent_hybrid*100)}Percent_TFLite'] = tflite_accuracy
              else:
                   print(f"{output_prefix} TFLite conversion failed.")
         else:
              print(f"{output_prefix} Keras model saving/fine-tuning failed.")
    else:
        print("Selective Hybrid QAT preparation failed.")

    # New: Middle Layers Selective QAT
    print(f"\n--- Processing Selective Middle Layers QAT (Exclude {MIDDLE_EXCLUDE_PERCENT*100:.0f}% Start/End) ---")
    base_model_for_middle_qat = tf.keras.models.clone_model(base_model)
    base_model_for_middle_qat.set_weights(base_model.get_weights()) # Copy weights
    qat_prepared_middle, actual_percent_middle = prepare_qat_middle_percent(base_model_for_middle_qat, exclude_percent=MIDDLE_EXCLUDE_PERCENT)
    if qat_prepared_middle is not None:
         output_prefix = f'resnet_middle_qat_{int(actual_percent_middle*100)}percent'
         saved_keras_qat_path, keras_accuracy_qat_middle = finetune_and_save_qat_keras_model( # Capture Keras accuracy
              qat_prepared_middle,
              finetune_data, finetune_labels, test_images, test_labels, output_prefix
         )
         if keras_accuracy_qat_middle is not None: # Store Keras accuracy
              evaluation_results[f'Selective_Middle_Layers_QAT_{int(actual_percent_middle*100)}Percent_Keras_FakeQuant'] = keras_accuracy_qat_middle

         # Convert Saved Keras model to TFLite
         if saved_keras_qat_path:
              tflite_path = convert_to_tflite(saved_keras_qat_path, f'{output_prefix}_qat.tflite')
              # Evaluate the TFLite model
              if tflite_path:
                   tflite_accuracy = evaluate_tflite(tflite_path, test_images, test_labels)
                   if tflite_accuracy is not None:
                        evaluation_results[f'Selective_Middle_Layers_QAT_{int(actual_percent_middle*100)}Percent_TFLite'] = tflite_accuracy
              else:
                   print(f"{output_prefix} TFLite conversion failed.")
         else:
              print(f"{output_prefix} Keras model saving/fine-tuning failed.")
    else:
        print("Selective Middle Layers QAT preparation failed.")


    # 4. Full INT8 PTQ Process (Prepare -> Convert -> Evaluate)
    print("\n--- Processing Full INT8 PTQ ---")
    base_model_for_ptq = tf.keras.models.clone_model(base_model)
    base_model_for_ptq.set_weights(base_model.get_weights()) # Copy weights
    # This function saves Keras temp and converts to TFLite
    ptq_tflite_path = quantize_activation_ptq(base_model_for_ptq)

    # Evaluate the TFLite model
    if ptq_tflite_path:
         tflite_accuracy = evaluate_tflite(ptq_tflite_path, test_images, test_labels)
         if tflite_accuracy is not None:
              evaluation_results['Full_INT8_PTQ_TFLite'] = tflite_accuracy
    else:
         print("Full INT8 PTQ conversion failed.")


    # --- Summary of Results ---
    print("\n--- Summary of Results ---")
    if evaluation_results:
        # Sort results for clearer output
        # Prioritize FP32, then Full QAT/PTQ TFLite, then Selective QAT TFLite, then Keras FakeQuant results
        def sort_key(item):
            name = item[0]
            if name == 'FP32_Base_Model': return 0
            if name == 'Full_INT8_PTQ_TFLite': return 1
            if name == 'Full_INT8_QAT_TFLite': return 2
            if name.endswith('_TFLite'): return 3 # All other TFLite results
            if name.endswith('_Keras_FakeQuant'): return 4 # Keras FakeQuant results
            return 5 # Fallback

        # Separate Keras and TFLite results for potentially clearer sorting/display
        keras_results = {k:v for k,v in evaluation_results.items() if k.endswith('_Keras_FakeQuant')}
        tflite_results = {k:v for k,v in evaluation_results.items() if k.endswith('_TFLite') or k == 'FP32_Base_Model'} # Include FP32 here


        print("\n--- TFLite Model Accuracy ---")
        if tflite_results:
             # Sort TFLite results (FP32 first, then Full, then Selective)
             def tflite_sort_key(item):
                 name = item[0]
                 if name == 'FP32_Base_Model': return 0
                 if name == 'Full_INT8_PTQ_TFLite': return 1
                 if name == 'Full_INT8_QAT_TFLite': return 2
                 # Sort selective TFLite results alphabetically
                 if name.endswith('_TFLite'): return 3 + ord(name[11]) # Sorts by first letter after Selective_
                 return 10 # Should not happen

             sorted_tflite_results = dict(sorted(tflite_results.items(), key=tflite_sort_key))
             for name, acc in sorted_tflite_results.items():
                  if acc is not None:
                      print(f"{name}: {acc*100:.2f}% Accuracy")
                  else:
                      print(f"{name}: Evaluation Failed")
        else:
            print("(No TFLite models evaluated successfully)")


        print("\n--- QAT Keras Model (Fake Quant) Accuracy ---")
        if keras_results:
             # Sort Keras FakeQuant results (Full first, then Selective by name)
             def keras_sort_key(item):
                  name = item[0]
                  if name == 'Full_INT8_QAT_Keras_FakeQuant': return 0
                  # Sort selective Keras results alphabetically
                  if name.endswith('_Keras_FakeQuant'): return 1 + ord(name[11]) # Sorts by first letter after Selective_
                  return 10 # Should not happen

             sorted_keras_results = dict(sorted(keras_results.items(), key=keras_sort_key))
             for name, acc in sorted_keras_results.items():
                  if acc is not None:
                      print(f"{name}: {acc*100:.2f}% Accuracy")
                  else:
                      print(f"{name}: Evaluation Failed")
        else:
            print("(No QAT Keras models evaluated successfully)")

    else:
        print("No models were successfully evaluated.")


print("\nComparison complete.")