import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import shutil

# --- Configuration for Additional Training ---
LOAD_MODEL_DIR = 'resnet_fp32_saved_model' # Directory to load the base model from
SAVE_MODEL_DIR = 'resnet_fp32_saved_model' # Directory to save the improved model to (overwrites)
ADDITIONAL_EPOCHS = 30 # Number of *additional* epochs to train
BATCH_SIZE = 500
LEARNING_RATE = 0.00005 # Use a lower learning rate for continued training
# Use the full dataset for training and testing during this phase
USE_FULL_TRAIN_DATA = True
USE_FULL_TEST_DATA = True

# --- Prepare Data ---
print("Loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

train_images = x_train if USE_FULL_TRAIN_DATA else x_train[:10000] # Use full or subset
train_labels = y_train if USE_FULL_TRAIN_DATA else y_train[:10000]

test_images = x_test if USE_FULL_TEST_DATA else x_test[:1000] # Use full or subset
test_labels = y_test if USE_FULL_TEST_DATA else y_test[:1000]


print(f"Using {len(train_images)} images for additional training.")
print(f"Using {len(test_images)} images for validation and final evaluation.")


# --- Load Base Model ---
print(f"\nLoading model from {LOAD_MODEL_DIR}...")
try:
    if not os.path.exists(LOAD_MODEL_DIR):
         print(f"Error: Model directory not found: {LOAD_MODEL_DIR}")
         print("Please ensure 'resnet_fp32_saved_model' exists and is a valid Keras SavedModel.")
         # Option to create a dummy model for testing if needed, but safer to exit
         exit() # Exit if model doesn't exist

    model = tf.keras.models.load_model(LOAD_MODEL_DIR)
    print("Successfully loaded base model.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Exiting.")
    exit()


# --- Compile Model ---
print("Compiling model for additional training...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
print("Model compiled.")


# --- Callbacks ---
# Use ModelCheckpoint to save the best model during training
# Create a temporary directory to save checkpoints
checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_filepath = os.path.join(checkpoint_dir, 'best_fp32_model')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,      # Save only the best model based on monitor
    monitor='val_accuracy',   # Monitor validation accuracy
    mode='max'                # Save when validation accuracy is maximized
)

# Optional: EarlyStopping if validation accuracy stops improving
# early_stopping_callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_accuracy',
#     patience=10, # Number of epochs with no improvement after which training will be stopped.
#     restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
# )

callbacks = [model_checkpoint_callback] # Add early_stopping_callback if desired


# --- Train Model ---
print(f"\nStarting additional training for {ADDITIONAL_EPOCHS} epochs...")
try:
    history = model.fit(
        train_images,
        train_labels,
        epochs=ADDITIONAL_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
        verbose=1 # Show training progress
    )
    print("Additional training complete.")

except Exception as e:
    print(f"Error during additional training: {e}")
    import traceback
    traceback.print_exc()
    print("Training aborted.")
    # Even if training aborts, try loading the best checkpoint if it exists
    pass


# --- Evaluate the Best Model ---
# Load the model that had the best validation accuracy
print(f"\nLoading best model from {checkpoint_filepath} for final evaluation...")
try:
    # Check if the checkpoint was actually saved
    if tf.io.gfile.exists(checkpoint_filepath):
         best_model = tf.keras.models.load_model(checkpoint_filepath)
         print("Successfully loaded best model checkpoint.")
    else:
         print("Warning: No best model checkpoint found. Using the model from the last epoch.")
         best_model = model # Use the model from the end of training

    # Ensure the model is compiled for evaluation if it wasn't restored with optimizer state
    # (ModelCheckpoint with save_best_only=True might save only weights, depending on TF version and save format)
    if not hasattr(best_model, 'optimizer'):
         print("Compiling best model for evaluation...")
         best_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    print(f"Evaluating the best model on {len(test_images)} test images...")
    loss, accuracy = best_model.evaluate(test_images, test_labels, verbose=0)
    print(f"Final Best Model Test Accuracy: {accuracy*100:.2f}% ({int(accuracy*len(test_images))}/{len(test_images)})")

except Exception as e:
    print(f"Error evaluating the best model: {e}")
    import traceback
    traceback.print_exc()
    print("Final evaluation failed.")
    # If evaluation failed, the script should probably not overwrite the saved model.
    # Let's just exit here or handle the error appropriately.
    exit() # Exit if evaluation fails


# --- Save the Improved Model ---
print(f"\nSaving the improved model to {SAVE_MODEL_DIR} (overwriting existing)...")
try:
    # Clean up the existing directory before saving
    if os.path.exists(SAVE_MODEL_DIR):
        print(f"Clearing existing directory: {SAVE_MODEL_DIR}")
        shutil.rmtree(SAVE_MODEL_DIR)

    best_model.save(SAVE_MODEL_DIR)
    print(f"Improved model saved successfully to {SAVE_MODEL_DIR}.")

except Exception as e:
    print(f"Error saving the improved model: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Clean up checkpoint directory
    if os.path.exists(checkpoint_dir):
        print(f"Cleaning up checkpoint directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)


print("\nAdditional training and saving complete.")
