import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.quantization
import torch.jit
import numpy as np
import os
import math # For ceiling function if needed
import collections # For OrderedDict

# --- Configuration ---
MODEL_SAVE_PATH_FP32 = 'resnet_fp32_pytorch.pth' # Placeholder path for dummy or real model
CALIB_SIZE = 100
TEST_SIZE = 1000
TARGET_QUANT_PERCENTAGE = 0.70 # Target 70% of parameters

# --- Define a Simple CNN Model (Similar to Keras Dummy for demonstration) ---
# Replace with your actual ResNet model definition if you have one
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False) # Bias=False for Conv-BN fusion
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU() # Moved ReLU out for potential fusion

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False) # Bias=False for Conv-BN fusion
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU() # Moved ReLU out for potential fusion

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # Flatten is implicitly handled by view or shape manipulation before Linear
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

    # Helper to list quantizable modules by name
    def get_quantizable_modules(self):
        quantizable_modules = collections.OrderedDict()
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                 # Only include modules with parameters
                 if any(p.numel() > 0 for p in module.parameters()):
                     quantizable_modules[name] = module
        return quantizable_modules

# --- Data Preparation ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR-10 Mean/Std
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use subsets for calibration and testing
calib_dataset = torch.utils.data.Subset(train_dataset, range(CALIB_SIZE))
test_dataset_subset = torch.utils.data.Subset(test_dataset, range(TEST_SIZE))

calib_loader = data.DataLoader(calib_dataset, batch_size=10, shuffle=False) # Batch size for calibration
test_loader = data.DataLoader(test_dataset_subset, batch_size=100, shuffle=False) # Batch size for testing

print(f"Using {len(calib_dataset)} calibration images and {len(test_dataset_subset)} test images.")


# --- Utility: Evaluate Model ---
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    # Ensure model is on the correct device
    model.to(device)

    # Check input/output types of the first batch for quantized models
    # PyTorch's standard static quantization often has float I/O wrappers
    # but the internal representation is quantized.
    # Direct evaluation of the returned 'converted' model is the most reliable.
    # We don't need special int8 handling like TFLite interpreter here
    # unless we explicitly save/load a fixed-point tensor format (like .ptl)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy

# --- Utility: Calibration Function ---
def calibrate_model(model, data_loader, device):
    print("Calibrating model...")
    model.eval()
    model.to(device)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    print("Calibration finished.")


# --- Helper Function for Parameter Counting ---
def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# --- Helper Function for Percentage-Based Selection (Adapted for PyTorch Modules) ---
def select_modules_by_percentage(model, score_dict, target_percentage=0.7, quantize_lowest=True):
    """
    Selects modules for quantization based on a score until a target percentage
    of parameters in quantizable modules is reached.

    Args:
        model: The PyTorch model (nn.Module).
        score_dict: A dictionary mapping module names (string) to their scores.
        target_percentage: The desired percentage of parameters to quantize (e.g., 0.7 for 70%).
        quantize_lowest: If True, sort scores ascending and quantize lowest scores first.
                         If False, sort descending and quantize highest scores first.

    Returns:
        A tuple containing:
          - set: A set of module names selected for quantization.
          - float: The actual percentage of parameters covered by the selection.
    """
    quantizable_modules_info = []
    total_quantizable_params = 0

    quantizable_modules = model.get_quantizable_modules() # Get Conv2d, Linear modules

    for name, module in quantizable_modules.items():
        # Check if score exists for this module
        if name in score_dict:
            params = count_parameters(module)
            if params > 0:
                quantizable_modules_info.append({
                    'name': name,
                    'score': score_dict[name],
                    'params': params
                })
                total_quantizable_params += params
        else:
            print(f"Warning: Score not found for quantizable module {name}. Skipping for selection consideration.")

    if total_quantizable_params == 0:
        print("No quantizable parameters found in Conv2d/Linear modules.")
        return set(), 0.0

    # Sort modules based on score and the quantize_lowest flag
    sorted_modules = sorted(quantizable_modules_info, key=lambda x: x['score'], reverse=not quantize_lowest)

    target_params = total_quantizable_params * target_percentage
    selected_modules_set = set()
    accumulated_params = 0

    print(f"\nTargeting {target_percentage*100:.1f}% ({int(target_params)}) of {total_quantizable_params} total quantizable parameters in quantizable modules.")
    print(f"Sorting modules by score ({'Ascending - quantizing lowest scores first' if quantize_lowest else 'Descending - quantizing highest scores first'})...")

    for module_info in sorted_modules:
        if accumulated_params < target_params:
            selected_modules_set.add(module_info['name'])
            accumulated_params += module_info['params']
            # print(f"  Selecting {module_info['name']} (Score: {module_info['score']:.4f}, Params: {module_info['params']}) -> Accumulated params: {accumulated_params}")
        else:
            # Stop once target is reached or exceeded
            # print(f"  Stopping selection. Target reached/exceeded.")
            break

    actual_percentage = (accumulated_params / total_quantizable_params) if total_quantizable_params > 0 else 0
    print(f"Selected {len(selected_modules_set)} modules for quantization.")
    print(f"Covered {accumulated_params} parameters ({actual_percentage*100:.2f}% of total quantizable parameters).")

    return selected_modules_set, actual_percentage

# --- PyTorch Quantization Helper ---

def quantize_model_selective(model, selected_module_names, save_path_prefix, device):
    """Helper to apply selective quantization based on module names."""
    print(f"Applying selective quantization to modules: {selected_module_names}")

    model.eval() # Ensure model is in evaluation mode
    model.to(device)

    # Clone the model to avoid modifying the original FP32 model directly
    model_to_quantize = SimpleCNN() # Or your ResNet class
    model_to_quantize.load_state_dict(model.state_dict())
    model_to_quantize.to(device)
    model_to_quantize.eval()

    # Define a standard static quantization config
    # 'fbgemm' is good for server/desktop CPU
    # 'qnnpack' is good for ARM CPU
    qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Set qconfig for modules. Set to None for modules NOT selected.
    # Also need to consider fusion. We should fuse first, then set qconfigs.

    # Identify fusible modules and fuse them
    # For SimpleCNN: Conv-BN-ReLU pattern
    # For ResNet: More complex patterns exist (Conv-BN-ReLU, Add-ReLU, Conv-BN, etc.)
    # You might need to adapt fusion patterns based on your model
    try:
        # List fusible layers by name in the model (adjust patterns for your model)
        # In SimpleCNN: conv1 -> bn1 -> relu1, conv2 -> bn2 -> relu2
        # Note: Fusion works best on a sequential list of module names.
        # The order should match the forward pass.
        fusable_patterns = []
        fusable_modules = []

        # SimpleCNN specific patterns
        if hasattr(model_to_quantize, 'conv1') and hasattr(model_to_quantize, 'bn1') and hasattr(model_to_quantize, 'relu1'):
             fusable_patterns.append(['conv1', 'bn1', 'relu1'])
        if hasattr(model_to_quantize, 'conv2') and hasattr(model_to_quantize, 'bn2') and hasattr(model_to_quantize, 'relu2'):
             fusable_patterns.append(['conv2', 'bn2', 'relu2'])

        if fusable_patterns:
             print(f"Attempting to fuse modules: {fusable_patterns}")
             # Need to fuse on the model instance
             fused_model = torch.quantization.fuse_modules(model_to_quantize, fusable_patterns)
             # Now, work with the fused model. The selected names need mapping if they were part of a fused group.
             # This mapping can be complex. A simpler approach for selective might be
             # to skip fusion or handle mapping explicitly.
             # For simplicity here, let's assume we set qconfig on *fused* modules.
             # A fused module name might be the name of the first module in the pattern.
             # This requires careful checking of the fused model's named_modules.

             # Let's rebuild a module name -> fused module name map after fusion
             name_map = {}
             for name, module in fused_model.named_modules():
                  # If a module was fused, its name might still be the first module name.
                  # This is tricky. A better approach for *selective* might be to fuse *first*
                  # then identify the names of the *resulting* fused modules or individual modules
                  # that are instances of quantizable types, and match those to original selected names.
                  # Let's simplify: Skip fusion for selective quantization for clarity,
                  # unless specifically selected modules require fusion (e.g. Conv-BN).

              # **Alternative Selective Strategy: Set qconfig directly without global fusion**
                print("Skipping global fusion for selective quantization.")
                model_for_q = model_to_quantize # Work on the cloned model directly

             # Create a default qconfig that includes observers for weights and activations
             # Use default_per_channel_weight_observer and default_observer
             qconfig = torch.quantization.QConfig(
                 activation=torch.quantization.observer.MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.qint8, reduce_range=True),
                 weight=torch.quantization.observer.MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, ch_axis=0, qscheme=torch.qint8, reduce_range=True)
             )


             # Set qconfig for selected modules, None for others
             for name, module in model_for_q.named_modules():
                 if name in selected_module_names:
                     if isinstance(module, (nn.Conv2d, nn.Linear)):
                         print(f"  -> Setting qconfig for {name}")
                         module.qconfig = qconfig
                     else:
                         # Handle cases where a selected name isn't directly quantizable (e.g., parent module)
                         # Or if the selected name is a module containing quantizable submodules.
                         # For simplicity, we assume selected_module_names are directly Conv/Linear names.
                         pass # qconfig is not set for non-quantizable layers anyway unless they are containers

                 else:
                     # Set qconfig to None for non-selected quantizable modules to prevent their quantization
                     if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)): # Include types that might be fused or need to be non-quantized
                         # Need to be careful: setting qconfig=None on containers might prevent inner modules
                         # Let's only explicitly set qconfig=None on quantizable types that are *not* selected.
                         if isinstance(module, (nn.Conv2d, nn.Linear)):
                             module.qconfig = None # Explicitly prevent quantization


             # Prepare the model - this inserts observers based on qconfig
             print("Preparing model with observers...")
             # Use torch.ao.quantization API for better control
             prepared_model = torch.ao.quantization.prepare(model_for_q)


             # Calibrate the model - observers collect statistics
             calibrate_model(prepared_model, calib_loader, device)

             # Convert the model - observers are used to compute scale/zero_point
             print("Converting model...")
             converted_model = torch.ao.quantization.convert(prepared_model)

             save_path_pt = f'{save_path_prefix}.pt'
             # Save the state dict or the model object
             torch.save(converted_model.state_dict(), save_path_pt)
             print(f"Saved quantized model state dict: {save_path_pt}")

             # Optional: Save as TorchScript for deployment (evaluation might differ slightly)
             # try:
             #     scripted_model = torch.jit.script(converted_model)
             #     script_save_path = f'{save_path_prefix}_scripted.pt'
             #     scripted_model.save(script_save_path)
             #     print(f"Saved TorchScript model: {script_save_path}")
             # except Exception as e:
             #     print(f"Warning: Could not script converted model: {e}")


             return converted_model # Return the in-memory converted model for direct evaluation


    except Exception as e:
        print(f"Error during selective quantization process for {save_path_prefix}: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Scoring Functions ---

# 1) Weight L2-norm based selective quantization
def quantize_l2_percent(model, target_percentage=0.7, device='cpu'):
    print(f"\n--- Starting L2 Norm Percentage Quantization ({target_percentage*100:.0f}%) ---")
    model.eval()
    model.to(device)
    module_scores = {}

    quantizable_modules = model.get_quantizable_modules()

    for name, module in quantizable_modules.items():
        weights = None
        if isinstance(module, nn.Conv2d):
            weights = module.weight
        elif isinstance(module, nn.Linear):
            weights = module.weight

        if weights is not None:
            try:
                l2_norm = torch.linalg.norm(weights.flatten()).item()
                module_scores[name] = l2_norm
                # print(f"Module {name}: L2 Norm = {l2_norm:.4f}")
            except Exception as e:
                print(f"Could not process module {name} for L2 norm: {e}")

    selected_module_names, actual_percentage = select_modules_by_percentage(
        model, module_scores, target_percentage, quantize_lowest=True # Quantize low L2 norm layers
    )

    if not selected_module_names:
        print("No modules selected for L2 norm quantization. Aborting.")
        return None

    save_path_prefix = f'resnet_l2norm_{int(actual_percentage*100)}percent_mpq'
    return quantize_model_selective(model, selected_module_names, save_path_prefix, device)


# 2) Activation-based full INT8 quantization (Post-Training Static Quantization)
def quantize_full_int8(model, save_path_prefix='resnet_activation_int8', device='cpu'):
    print("\n--- Starting Full INT8 Post-Training Static Quantization ---")
    model.eval()
    model.to(device)

    # Clone the model for quantization
    model_to_quantize = SimpleCNN() # Or your ResNet class
    model_to_quantize.load_state_dict(model.state_dict())
    model_to_quantize.to(device)
    model_to_quantize.eval()

    # Define Fusion patterns (adjust for your model)
    fusable_patterns = []
    if hasattr(model_to_quantize, 'conv1') and hasattr(model_to_quantize, 'bn1') and hasattr(model_to_quantize, 'relu1'):
         fusable_patterns.append(['conv1', 'bn1', 'relu1'])
    if hasattr(model_to_quantize, 'conv2') and hasattr(model_to_quantize, 'bn2') and hasattr(model_to_quantize, 'relu2'):
         fusable_patterns.append(['conv2', 'bn2', 'relu2'])
    # Add other patterns if necessary for your model (e.g., 'fc', 'relu') - Linear-ReLU is common

    if fusable_patterns:
        print(f"Attempting to fuse modules: {fusable_patterns}")
        try:
             model_to_quantize = torch.quantization.fuse_modules(model_to_quantize, fusable_patterns)
             print("Fusion successful.")
        except Exception as e:
             print(f"Fusion failed: {e}. Proceeding without fusion.")


    # Define the quantization configuration - static quantization
    # 'fbgemm' for server/desktop CPU, 'qnnpack' for ARM CPU
    qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Apply qconfig to the entire model - this means all supported modules will be quantized
    model_to_quantize.qconfig = qconfig
    print("QConfig applied.")

    # Prepare the model for quantization - this inserts observers
    # Use torch.ao.quantization API
    prepared_model = torch.ao.quantization.prepare(model_to_quantize)
    print("Model prepared with observers.")

    # Calibrate the model - observers collect statistics on activation ranges
    calibrate_model(prepared_model, calib_loader, device)

    # Convert the model - observers are used to compute scale/zero_point, modules are replaced with quantized versions
    print("Converting model...")
    converted_model = torch.ao.quantization.convert(prepared_model)
    print("Conversion finished.")

    save_path_pt = f'{save_path_prefix}.pt'
    # Save the state dict
    torch.save(converted_model.state_dict(), save_path_pt)
    print(f"Saved quantized model state dict: {save_path_pt}")

    # Optional: Save as TorchScript
    # try:
    #     scripted_model = torch.jit.script(converted_model)
    #     script_save_path = f'{save_path_prefix}_scripted.pt'
    #     scripted_model.save(script_save_path)
    #     print(f"Saved TorchScript model: {script_save_path}")
    # except Exception as e:
    #      print(f"Warning: Could not script converted model: {e}")

    return converted_model # Return the in-memory converted model for direct evaluation


# 3) Hessian-based sensitivity (Percentage)
# Hessian calculation in PyTorch
def calculate_sensitivity_scores(model, data_loader, device):
    print("Calculating Hessian-based Sensitivity Scores...")
    model.eval()
    model.to(device)
    module_sens_scores = {}
    loss_fn = nn.CrossEntropyLoss()

    # Use only one batch for approximation as in the TF code
    inputs, labels = next(iter(data_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    model.zero_grad() # Zero out gradients from previous steps

    # Enable gradients for parameters of quantizable modules
    # By default, gradients are often True if requires_grad is set during module creation,
    # but explicitly set/check for safety if needed.

    # Forward pass and loss calculation
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

    # Backward pass to compute gradients
    loss.backward()

    # Calculate sensitivity score for each parameter and aggregate by module
    quantizable_modules = model.get_quantizable_modules()

    for name, module in quantizable_modules.items():
        module_score = 0.0
        for param_name, param in module.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Approximation: Mean of (Grad * Parameter)^2
                score = torch.mean(torch.square(param.grad * param)).item()
                module_score += score # Aggregate scores for module's parameters
                # print(f"  Param {name}.{param_name}: Score part = {score:.6f}")

        module_sens_scores[name] = module_score

    print("Sensitivity score calculation finished.")
    # print("Sensitivity scores:", module_sens_scores)
    return module_sens_scores


def quantize_hessian_percent(model, target_percentage=0.7, device='cpu'):
    print(f"\n--- Starting Hessian (Gradient Approx) Percentage Quantization ({target_percentage*100:.0f}%) ---")
    module_sens = calculate_sensitivity_scores(model, calib_loader, device)

    if not module_sens:
        print("Could not calculate sensitivity scores. Aborting Hessian quantization.")
        return None

    selected_module_names, actual_percentage = select_modules_by_percentage(
        model, module_sens, target_percentage, quantize_lowest=True # Quantize low sensitivity modules
    )

    if not selected_module_names:
        print("No modules selected for Hessian quantization. Aborting.")
        return None

    save_path_prefix = f'resnet_hessian_{int(actual_percentage*100)}percent_mpq'
    return quantize_model_selective(model, selected_module_names, save_path_prefix, device)


# 4) Hybrid: combine L2 norm + activation range + sensitivity (Percentage)
# Activation Range calculation using forward hooks
class ActivationRangeHook:
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def __call__(self, module, input, output):
        # Ensure output is a tensor and is numeric
        if isinstance(output, torch.Tensor) and output.is_floating_point():
            self.min_val = min(self.min_val, torch.min(output).item())
            self.max_val = max(self.max_val, torch.max(output).item())
        # Handle fused modules which might output tuple (tensor, tensor)
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor) and output[0].is_floating_point():
             self.min_val = min(self.min_val, torch.min(output[0]).item())
             self.max_val = max(self.max_val, torch.max(output[0]).item())


def calculate_activation_ranges(model, data_loader, device):
    print("Calculating Activation Ranges...")
    model.eval()
    model.to(device)
    hooks = {}
    module_ranges = {}

    quantizable_modules = model.get_quantizable_modules()

    # Register hooks to capture activation ranges
    for name, module in quantizable_modules.items():
        hook = ActivationRangeHook()
        # Register hook on the module itself to capture its output
        hooks[name] = module.register_forward_hook(hook)

    # Run calibration data through the model
    with torch.no_grad():
         for inputs, labels in data_loader:
             inputs = inputs.to(device)
             _ = model(inputs)

    # Collect results and remove hooks
    for name, hook in hooks.items():
        hook.remove() # Remove the hook

        # Calculate range, handle inf values if no data passed through the module
        min_val = hook.min_val if hook.min_val != float('inf') else 0
        max_val = hook.max_val if hook.max_val != float('-inf') else 0
        module_ranges[name] = max_val - min_val

    print("Activation range calculation finished.")
    # print("Activation ranges:", module_ranges)
    return module_ranges


def quantize_hybrid_percent(model, alpha=0.4, beta=0.4, gamma=0.2, target_percentage=0.7, device='cpu'):
    print(f"\n--- Starting Hybrid Percentage Quantization ({target_percentage*100:.0f}%) ---")
    model.eval()
    model.to(device)

    # Calculate component scores
    l2_scores = {}
    quantizable_modules = model.get_quantizable_modules()
    for name, module in quantizable_modules.items():
        weights = None
        if isinstance(module, nn.Conv2d):
            weights = module.weight
        elif isinstance(module, nn.Linear):
            weights = module.weight
        l2_scores[name] = torch.linalg.norm(weights.flatten()).item() if weights is not None else 0

    act_ranges = calculate_activation_ranges(model, calib_loader, device)
    sens_scores = calculate_sensitivity_scores(model, calib_loader, device)


    # Normalize scores (handle potential division by zero if max is 0)
    max_l2 = max(l2_scores.values()) if l2_scores else 1.0
    max_range = max(act_ranges.values()) if act_ranges else 1.0
    max_sens = max(sens_scores.values()) if sens_scores else 1.0

    hybrid_scores = {}
    print("\nCalculating Hybrid Scores:")
    for name in quantizable_modules.keys(): # Iterate through actual quantizable modules
        norm_l2 = (l2_scores.get(name, 0) / max_l2) if max_l2 != 0 else 0
        norm_range = (act_ranges.get(name, 0) / max_range) if max_range != 0 else 0
        norm_sens = (sens_scores.get(name, 0) / max_sens) if max_sens != 0 else 0

        # Combine scores - assuming lower is better for quantization
        score = alpha * norm_l2 + beta * norm_range + gamma * norm_sens
        hybrid_scores[name] = score
        # print(f"  Module {name}: NormL2={norm_l2:.3f}, NormRange={norm_range:.3f}, NormSens={norm_sens:.3f} -> Hybrid Score = {score:.4f}")

    # Select modules based on the lowest hybrid scores
    selected_module_names, actual_percentage = select_modules_by_percentage(
        model, hybrid_scores, target_percentage, quantize_lowest=True
    )

    if not selected_module_names:
        print("No modules selected for Hybrid quantization. Aborting.")
        return None

    save_path_prefix = f'resnet_hybrid_{int(actual_percentage*100)}percent_mpq'
    return quantize_model_selective(model, selected_module_names, save_path_prefix, device)


# --- Main Execution ---
if __name__ == "__main__":
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Base Model ---
    base_model = SimpleCNN() # Instantiate your model class

    # Try loading a pre-trained state dict
    if os.path.exists(MODEL_SAVE_PATH_FP32):
        try:
            base_model.load_state_dict(torch.load(MODEL_SAVE_PATH_FP32))
            print(f"Successfully loaded base model from {MODEL_SAVE_PATH_FP32}.")
        except Exception as e:
            print(f"Error loading model from {MODEL_SAVE_PATH_FP32}: {e}")
            print("Using randomly initialized model.")
    else:
        print(f"Model file not found at {MODEL_SAVE_PATH_FP32}. Using randomly initialized model and saving it.")
        # Save the randomly initialized model state dict for future runs
        try:
             torch.save(base_model.state_dict(), MODEL_SAVE_PATH_FP32)
             print(f"Saved randomly initialized model to {MODEL_SAVE_PATH_FP32}.")
        except Exception as e:
             print(f"Error saving dummy model: {e}")


    # Evaluate FP32 model baseline
    print("\n--- Evaluating FP32 Base Model ---")
    # Ensure model is on device before evaluating
    base_model.to(device)
    fp32_accuracy = evaluate_model(base_model, test_loader, device)
    print(f"FP32 Base Model Accuracy: {fp32_accuracy*100:.2f}%")


    # --- Perform Quantization ---
    print("\nStarting Quantization Procedures...")

    quantized_models = {}

    # Full INT8
    quantized_models['Full_INT8'] = quantize_full_int8(base_model, device=device)

    # Selective Quantization methods
    quantized_models['L2_Norm_Percent'] = quantize_l2_percent(base_model, target_percentage=TARGET_QUANT_PERCENTAGE, device=device)
    quantized_models['Hessian_Percent'] = quantize_hessian_percent(base_model, target_percentage=TARGET_QUANT_PERCENTAGE, device=device)
    quantized_models['Hybrid_Percent'] = quantize_hybrid_percent(base_model, target_percentage=TARGET_QUANT_PERCENTAGE, device=device)


    # --- Evaluate Quantized Models ---
    print("\nStarting Evaluations of Quantized Models...")

    # Filter out methods that failed to produce a model
    quantized_models = {name: model for name, model in quantized_models.items() if model is not None}

    for name, model in quantized_models.items():
        print(f"\n--- Evaluating Quantized Model: {name} ---")
        # Ensure the quantized model is on the correct device for evaluation
        model.to(device)
        evaluate_model(model, test_loader, device)

    print("\nQuantization and evaluation complete.")
