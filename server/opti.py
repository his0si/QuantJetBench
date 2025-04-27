import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.datasets import cifar10
import os
import shutil # Added for cleaning up directories

# --- Configuration ---
FINETUNE_SIZE = 10000 # Increased: Use 10000 images for QAT fine-tuning (closer to full CIFAR)
CALIB_SIZE = 1000     # Increased: Use 1000 images for PTQ and post-QAT TFLite calibration
TEST_SIZE = 1000      # Keep test size reasonable for quick evaluation
QAT_EPOCHS = 20       # Increased: Number of epochs for QAT fine-tuning (suggested >= 20)
TARGET_QUANT_PERCENTAGE = 0.70 # Target 70% of parameters for selective QAT
QAT_LEARNING_RATE = 1e-4 # Reduced: Lower learning rate for QAT fine-tuning (suggested 1e-4 or 1e-5)

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
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Use a subset of training data for QAT fine-tuning
# Ensure FINETUNE_SIZE + CALIB_SIZE does not exceed x_train length
if FINETUNE_SIZE + CALIB_SIZE > len(x_train):
    print(f"Warning: FINETUNE_SIZE ({FINETUNE_SIZE}) + CALIB_SIZE ({CALIB_SIZE}) exceeds total training data size ({len(x_train)}). Adjusting sizes.")
    FINETUNE_SIZE = len(x_train) - CALIB_SIZE # Maximize finetune size while keeping calib_size distinct
    if FINETUNE_SIZE < 0: # If calib_size is already larger than total train data
         CALIB_SIZE = len(x_train)
         FINETUNE_SIZE = 0
         print("Adjusted calib_size to total training data size, finetune_size is 0.")


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
        return False

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
        return True
    except Exception as e:
        print(f"Error during TFLite conversion for {output_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- Helper: Fine-tune QAT model and convert to TFLite ---
def finetune_and_convert_qat_model(qat_prepared_model, finetune_data, finetune_labels, test_data_eval, test_labels_eval, output_prefix):
    """Fine-tunes a QAT-prepared model and converts it to TFLite."""
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
    print("Unfreezing BatchNormalization layers...")
    for layer in qat_prepared_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            #print(f" - Unfrozen BN layer: {layer.name}") # Optional print


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
        # You might want to print/log history.history['accuracy'][-1] etc.
    except Exception as e:
        print(f"Error during QAT fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return None # Return None if fine-tuning fails


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
    except Exception as e:
        print(f"Error saving fine-tuned QAT model: {e}")
        # Clean up potentially partially created directory
        if os.path.exists(saved_qat_model_dir):
             shutil.rmtree(saved_qat_model_dir)
        return None # Return None if saving fails

    # Convert the fine-tuned QAT Keras model to TFLite (Full INT8)
    tflite_qat_path = f'{output_prefix}_qat.tflite' # Add _qat suffix
    success = convert_to_tflite(saved_qat_model_dir, tflite_qat_path)

    # Clean up the saved Keras model directory after conversion (optional but good practice)
    if os.path.exists(saved_qat_model_dir):
         try:
             shutil.rmtree(saved_qat_model_dir)
             # print(f"Cleaned up saved Keras model directory: {saved_qat_model_dir}")
         except Exception as clean_e:
             print(f"Warning: Could not clean up saved Keras model directory {saved_qat_model_dir}: {clean_e}")


    if success:
         return tflite_qat_path # Return the path to the created TFLite file
    else:
         return None

# --- Helper Function for Percentage-Based Selection (Brought back) ---
def select_layers_by_percentage(model, score_dict, target_percentage=0.7, quantize_lowest=True):
    """
    Selects layers for quantization based on a score until a target percentage
    of parameters is reached.
    """
    quantizable_layers = []
    total_quantizable_params = 0

    # Identify layers that are typically quantizable and have parameters
    quantizable_types = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)

    for layer in model.layers:
        if isinstance(layer, quantizable_types):
             params = layer.count_params()
             # Only consider layers with parameters
             if params > 0:
                  # Check if score exists for this layer. If not, it won't be considered for selection based on score.
                  if layer.name in score_dict:
                       quantizable_layers.append({
                           'name': layer.name,
                           'score': score_dict[layer.name],
                           'params': params
                       })
                       total_quantizable_params += params
                  # else: Warning about missing score is handled in scoring functions


    if total_quantizable_params == 0:
        print("No quantizable parameters with scores found for selection.")
        return set(), 0.0

    # Sort layers based on score and the quantize_lowest flag
    sorted_layers = sorted(quantizable_layers, key=lambda x: x['score'], reverse=not quantize_lowest)

    target_params = total_quantizable_params * target_percentage
    selected_layers_set = set()
    accumulated_params = 0

    print(f"\nTargeting {target_percentage*100:.1f}% ({int(target_params)}) of {total_quantizable_params} total quantizable parameters with scores.")
    print(f"Sorting layers by score ({'Ascending - quantizing lowest scores first' if quantize_lowest else 'Descending - quantizing highest scores first'})...")

    for layer_info in sorted_layers:
        if accumulated_params < target_params:
            selected_layers_set.add(layer_info['name'])
            accumulated_params += layer_info['params']
            # print(f"  Selecting {layer_info['name']} (Score: {layer_info['score']:.4f}, Params: {layer_info['params']}) -> Accumulated params: {accumulated_params}")
        else:
             # Stop once target is reached or exceeded, but the last layer added might exceed the target
             break

    actual_percentage = (accumulated_params / total_quantizable_params) if total_quantizable_params > 0 else 0
    print(f"Selected {len(selected_layers_set)} layers for QAT based on percentage target.")
    print(f"Covered {accumulated_params} parameters ({actual_percentage*100:.2f}% of total quantizable parameters with scores).")

    return selected_layers_set, actual_percentage

# --- Helper to get the annotated model for selective QAT (Brought back) ---
def get_annotated_model_selective(model, selected_layer_names):
     QuantAnnotate = tfmot.quantization.keras.quantize_annotate_layer
     def apply_quantization_based_on_selection(layer):
         # Check if the layer is quantizable and in the selected list
         quantizable_types = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)
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


# --- Prepare Full QAT Model (Corrected) ---
def prepare_qat_full(model):
    """Applies QAT preparation to the entire model using quantize_model."""
    print(f"\n--- Starting Full QAT Preparation ---")

    # Use quantize_model to automatically annotate and apply quantization to all supported layers
    with tfmot.quantization.keras.quantize_scope():
        # quantize_model handles annotation and applying quantization
        qat_model = tfmot.quantization.keras.quantize_model(model)

    print("Full QAT model prepared.")
    # For Full QAT, the percentage is effectively 100% of quantizable layers
    # Count total quantizable parameters in the original model for the 100% value
    total_quantizable_params = 0
    quantizable_types = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)
    for layer in model.layers:
         if isinstance(layer, quantizable_types):
              total_quantizable_params += layer.count_params() if layer.count_params() > 0 else 0

    # Return the model and 1.0 for 100% (and total params if needed, though not used in main)
    return qat_model, 1.0


# --- Scoring Functions (Brought back and integrated into prepare functions) ---
# Define common quantizable types for score calculation and selection consistency
QUANTIZABLE_SCORE_TYPES = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)


# 1) Weight L2-norm based selective QAT (Percentage)
def prepare_qat_l2_percent(model, target_percentage=0.7):
    print(f"\n--- Starting L2 Norm Percentage Selective QAT Preparation ({target_percentage*100:.0f}%) ---")
    layer_scores = {}
    # quantizable_layer_names = set() # Track layers we tried to score

    for layer in model.layers:
        if isinstance(layer, QUANTIZABLE_SCORE_TYPES):
            # quantizable_layer_names.add(layer.name) # Not strictly needed here if score_dict build handles it
            weights = layer.get_weights()
            if weights and len(weights) > 0 and weights[0] is not None:
                try:
                    l2_norm = np.linalg.norm(weights[0].flatten())
                    layer_scores[layer.name] = l2_norm
                except Exception as e:
                    print(f"Could not process layer {layer.name} for L2 norm: {e}")
            # else: Layers without weights[0] won't get a score and are implicitly skipped by select_layers


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
    # quantizable_layer_names = set() # Not strictly needed here
    param_to_layer = {} # Map variable references to layer names

    for layer in model.layers:
        if isinstance(layer, QUANTIZABLE_SCORE_TYPES):
            # quantizable_layer_names.add(layer.name) # Not strictly needed here
            for param in layer.trainable_variables: # Get trainable variables *for this layer*
                 param_to_layer[param.ref()] = layer.name # Use variable reference as key


    if not param_to_layer:
         print("No trainable variables in quantizable layers found to calculate Hessian scores.")
         return None, None


    try:
        print(f"Calculating gradients with input shape: {sample_images.shape}")
        # Ensure the model is built before calculating gradients
        _ = model(tf.random.uniform(shape=(1,) + model.input_shape[1:])) # Build the model if not already built

        with tf.GradientTape() as tape:
            # Watch all trainable variables explicitly to be safe
            # for var in model.trainable_variables:
            #     tape.watch(var) # This can sometimes slow things down or be unnecessary
            preds = model(sample_images, training=False) # Use training=False for inference path gradients
            loss = loss_fn(sample_labels, preds)
        print(f"Calculated loss: {loss.numpy()}")

        # Get gradients for all trainable variables in the model
        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        if grads is None or not any(g is not None for g in grads):
            print("Error: Gradients are None. Cannot calculate sensitivity. Check model, loss, data.")
            return None, None

        # Aggregate scores by module name
        for var, grad in zip(trainable_vars, grads):
            if grad is None: continue # Skip if gradient is None

            layer_name = param_to_layer.get(var.ref()) # Use the pre-built map
            # Ensure the variable belongs to a quantizable layer type we care about and has a mapping
            if layer_name is not None and layer_name in {l.name for l in model.layers if isinstance(l, QUANTIZABLE_SCORE_TYPES)}:
                 # Approximation: Mean of (Grad * Parameter)^2
                 # Ensure grad and var have compatible shapes or broadcast
                 # If grad/var are tensors, tf.square(grad * var) works fine
                 score = tf.reduce_mean(tf.square(grad * var)).numpy()
                 # Aggregate score by layer name (a layer might have weight and bias)
                 layer_sens[layer_name] = layer_sens.get(layer_name, 0) + score
            # else: print(f"Warning: Variable {var.name} not mapped to a quantizable layer.")


        # Ensure all potential quantizable layers have a score entry, even if 0
        all_potential_quantizable_names = {l.name for l in model.layers if isinstance(l, QUANTIZABLE_SCORE_TYPES)}
        for name in all_potential_quantizable_names:
            if name not in layer_sens:
                 layer_sens[name] = 0 # Assign 0 if no variables or grad found for this layer


        if not layer_sens:
            print("Error: Sensitivity scores dictionary is empty.")
            return None, None

        selected_layers, actual_percentage = select_layers_by_percentage(
            model, layer_sens, target_percentage, quantize_lowest=True # Quantize low sensitivity layers
        )

        if not selected_layers:
            print("No layers selected for Hessian QAT. Aborting preparation.")
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
    for layer in model.layers:
         if isinstance(layer, QUANTIZABLE_SCORE_TYPES):
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

        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        if grads is None or not any(g is not None for g in grads):
            print("Warning: Gradients are None for sensitivity calculation in hybrid.")
        else:
            # Aggregate scores by module name using the pre-built map
            for var, grad in zip(trainable_vars, grads):
                if grad is None: continue
                layer_name = param_to_layer.get(var.ref()) # Use the map
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
        print("No layers selected for Hybrid QAT. Aborting preparation.")
        return None, None

    annotated_model = get_annotated_model_selective(model, selected_layers)

    with tfmot.quantization.keras.quantize_scope():
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    print("QAT model prepared (Hybrid Percentage).")
    return qat_model, actual_percentage


# --- Full INT8 Post-Training Quantization (Baseline - Keep as is) ---
def quantize_activation_ptq(model):
    print("\n--- Starting Full INT8 Post-Training Quantization (PTQ Baseline) ---")
    save_path = 'model_activation_int8_ptq_keras_temp'
    tflite_path = 'resnet_activation_int8_ptq.tflite'
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
        success = convert_to_tflite(save_path, tflite_path)

        # Clean up temporary Keras model directory after conversion (optional but good practice)
        if os.path.exists(save_path):
             try:
                 shutil.rmtree(save_path)
                 # print(f"Cleaned up temporary directory: {save_path}")
             except Exception as clean_e:
                 print(f"Error cleaning up temporary directory {save_path}: {clean_e}")


        if success:
            return tflite_path
        else:
            return None
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
             model.compile(optimizer='adam', loss='sparse_categoricalCrossentropy', metrics=['accuracy'])


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

    # Dictionary to store paths of generated TFLite files for evaluation
    generated_tflite_files = {}
    evaluation_results = {} # Store evaluation results by name

    # 1. Evaluate Original FP32 Model
    fp32_accuracy = evaluate_keras_model(base_model, test_images, test_labels)
    if fp32_accuracy is not None:
        evaluation_results['FP32_Base_Model'] = fp32_accuracy

    # 2. Prepare, Fine-tune, and Convert Full QAT Model
    # Need to clone the base model before applying quantize_model to avoid modifying the original
    print("\n--- Processing Full QAT ---")
    base_model_for_full_qat = tf.keras.models.clone_model(base_model)
    base_model_for_full_qat.set_weights(base_model.get_weights()) # Copy weights

    # Call the corrected prepare_qat_full
    qat_prepared_full, actual_percent_full = prepare_qat_full(base_model_for_full_qat) # Pass the cloned model

    if qat_prepared_full is not None:
         # Use actual percentage (should be near 100) in filename
         output_prefix = f'resnet_full_qat_{int(actual_percent_full*100)}percent'
         tflite_path = finetune_and_convert_qat_model(
             qat_prepared_full,
             finetune_data,
             finetune_labels,
             test_images, # Pass test data for validation during fit
             test_labels, # Pass test labels for validation during fit
             output_prefix
         )
         if tflite_path:
             generated_tflite_files['Full_INT8_QAT'] = tflite_path
         else:
             print("Full QAT TFLite conversion failed.")
    else:
        print("Full QAT preparation failed.")


    # 3. Prepare, Fine-tune, and Convert Selective QAT Models
    print(f"\nStarting Selective QAT Procedures with Target Percentage: {TARGET_QUANT_PERCENTAGE*100:.0f}%")

    # L2 Norm Selective QAT
    # Clone the base model for each selective method to start fresh
    print("\n--- Processing Selective L2 Norm QAT ---")
    base_model_for_l2_qat = tf.keras.models.clone_model(base_model)
    base_model_for_l2_qat.set_weights(base_model.get_weights()) # Copy weights
    qat_prepared_l2, actual_percent_l2 = prepare_qat_l2_percent(base_model_for_l2_qat, target_percentage=TARGET_QUANT_PERCENTAGE)
    if qat_prepared_l2 is not None:
         output_prefix = f'resnet_l2norm_qat_{int(actual_percent_l2*100)}percent'
         tflite_path = finetune_and_convert_qat_model(
             qat_prepared_l2,
             finetune_data,
             finetune_labels,
             test_images, # Pass test data for validation during fit
             test_labels, # Pass test labels for validation during fit
             output_prefix
         )
         if tflite_path:
             generated_tflite_files[f'Selective_L2_QAT_{int(actual_percent_l2*100)}Percent'] = tflite_path
         else:
             print("Selective L2 QAT TFLite conversion failed.")
    else:
        print("Selective L2 QAT preparation failed.")

    # Hessian Selective QAT
    print("\n--- Processing Selective Hessian QAT ---")
    base_model_for_hessian_qat = tf.keras.models.clone_model(base_model)
    base_model_for_hessian_qat.set_weights(base_model.get_weights()) # Copy weights
    qat_prepared_hessian, actual_percent_hessian = prepare_qat_hessian_percent(base_model_for_hessian_qat, target_percentage=TARGET_QUANT_PERCENTAGE)
    if qat_prepared_hessian is not None:
         output_prefix = f'resnet_hessian_qat_{int(actual_percent_hessian*100)}percent'
         tflite_path = finetune_and_convert_qat_model(
             qat_prepared_hessian,
             finetune_data,
             finetune_labels,
             test_images, # Pass test data for validation during fit
             test_labels, # Pass test labels for validation during fit
             output_prefix
         )
         if tflite_path:
             generated_tflite_files[f'Selective_Hessian_QAT_{int(actual_percent_hessian*100)}Percent'] = tflite_path
         else:
             print("Selective Hessian QAT TFLite conversion failed.")
    else:
        print("Selective Hessian QAT preparation failed.")

    # Hybrid Selective QAT
    print("\n--- Processing Selective Hybrid QAT ---")
    base_model_for_hybrid_qat = tf.keras.models.clone_model(base_model)
    base_model_for_hybrid_qat.set_weights(base_model.get_weights()) # Copy weights
    qat_prepared_hybrid, actual_percent_hybrid = prepare_qat_hybrid_percent(base_model_for_hybrid_qat, target_percentage=TARGET_QUANT_PERCENTAGE)
    if qat_prepared_hybrid is not None:
         output_prefix = f'resnet_hybrid_qat_{int(actual_percent_hybrid*100)}percent'
         tflite_path = finetune_and_convert_qat_model(
             qat_prepared_hybrid,
             finetune_data,
             finetune_labels,
             test_images, # Pass test data for validation during fit
             test_labels, # Pass test labels for validation during fit
             output_prefix
         )
         if tflite_path:
             generated_tflite_files[f'Selective_Hybrid_QAT_{int(actual_percent_hybrid*100)}Percent'] = tflite_path
         else:
             print("Selective Hybrid QAT TFLite conversion failed.")
    else:
        print("Selective Hybrid QAT preparation failed.")


    # 4. Prepare and Convert Full INT8 PTQ Model (Baseline)
    # Clone the base model for PTQ to avoid any potential side effects
    print("\n--- Processing Full INT8 PTQ ---")
    base_model_for_ptq = tf.keras.models.clone_model(base_model)
    base_model_for_ptq.set_weights(base_model.get_weights()) # Copy weights
    ptq_tflite_path = quantize_activation_ptq(base_model_for_ptq) # Pass the cloned model

    if ptq_tflite_path:
         generated_tflite_files['Full_INT8_PTQ'] = ptq_tflite_path # Add PTQ result to evaluation list
    else:
        print("Full INT8 PTQ conversion failed.")


    # --- Evaluate all generated TFLite Models ---
    print("\nStarting Evaluations of Generated TFLite Models...")
    if generated_tflite_files:
        for name, tflite_path in generated_tflite_files.items():
             # Evaluate TFLite models and store accuracy in evaluation_results
             # Accuracy for TFLite models is already calculated and printed by evaluate_tflite
             # We just call it here to perform the evaluation for each file
             accuracy = evaluate_tflite(tflite_path, test_images, test_labels)
             if accuracy is not None:
                 # Use the name from generated_tflite_files dictionary
                 evaluation_results[name] = accuracy
             else:
                 evaluation_results[name] = None # Mark as failed evaluation
    else:
        print("No TFLite models were successfully generated for evaluation.")


    print("\n--- Summary of Results ---")
    if evaluation_results:
        # Sort results for cleaner output
        # Prioritize FP32, then Full QAT/PTQ, then Selective QATs by name
        def sort_key(item):
            name = item[0]
            if name == 'FP32_Base_Model': return 0
            if name == 'Full_INT8_PTQ': return 1
            if name == 'Full_INT8_QAT': return 2
            # Use a tuple for selective models to sort by percentage if needed, or just alphabetically
            # Example: ('Selective_L2_QAT_71Percent', 3)
            if name.startswith('Selective_'): return 3
            return 4 # Fallback for unexpected names

        sorted_results = dict(sorted(evaluation_results.items(), key=sort_key))

        for name, acc in sorted_results.items():
             if acc is not None:
                 print(f"{name}: {acc*100:.2f}% Accuracy")
             else:
                 print(f"{name}: Evaluation Failed")
    else:
        print("No models were successfully evaluated.")


    print("\nComparison complete.")