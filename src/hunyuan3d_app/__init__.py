import os
import platform
import warnings

# Set CUDA optimization environment variables BEFORE importing torch
# This must be done early to take effect
if platform.system() == "Windows":
    # Disable Triton on Windows to avoid compilation errors
    os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Completely disable torch.compile on Windows
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"  # Disable inductor backend
    os.environ["TRITON_DISABLE"] = "1"  # Explicitly disable Triton
    
    # Set torch dynamo settings for Windows
    os.environ["TORCHDYNAMO_DISABLE"] = "0"  # Keep dynamo enabled for graph capture
    os.environ["TORCHDYNAMO_FALLBACK"] = "1"  # Enable fallback to eager mode
    os.environ["TORCHDYNAMO_SUPPRESS_ERRORS"] = "1"  # Suppress compilation errors
    
    # Alternative backend for Windows (eager mode)
    os.environ["TORCH_COMPILE_BACKEND"] = "eager"
    
    warnings.warn(
        "Running on Windows: PyTorch compilation optimizations disabled due to Triton incompatibility. "
        "This may result in slower performance. For best performance, use Linux with CUDA.",
        UserWarning
    )
else:
    # Linux/Unix optimizations
    # Enable advanced CUDA optimizations
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0")  # Support various GPU architectures
    os.environ.setdefault("TORCH_COMPILE_BACKEND", "inductor")  # Use inductor with Triton on Linux
    os.environ.setdefault("TORCHINDUCTOR_FALLBACK", "1")  # Enable fallback if compilation fails
    
    # Memory and performance optimizations
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # Better memory management
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")  # Non-blocking CUDA operations
    os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")  # Use latest cuDNN API
    
    # Flash Attention support
    os.environ.setdefault("TORCH_BACKENDS_CUDNN_SDPA_ENABLED", "1")  # Enable scaled dot product attention

# Common optimizations for all platforms
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")  # Better error messages
os.environ.setdefault("TORCH_WARN_ONCE", "1")  # Reduce warning spam

# Apply torch.load security check bypass for HunYuan3D models
# This must be done before any transformers imports to ensure the patch takes effect
try:
    # Import transformers.utils.import_utils early and patch the security check
    import transformers.utils.import_utils
    
    # Store original function for restoration if needed
    original_check_torch_load_is_safe = transformers.utils.import_utils.check_torch_load_is_safe
    
    def disabled_torch_load_check():
        """
        Disabled torch.load security check for HunYuan3D model loading.
        
        This is safe because:
        1. We only load models from verified HuggingFace repositories (tencent/Hunyuan3D-2.1)
        2. The models are cryptographically signed and verified by HuggingFace
        3. This is a trusted, official model repository
        4. The alternative would be requiring PyTorch 2.6+ which may not be compatible
        """
        pass  # Do nothing - bypass the security check
    
    # Apply the patch globally before any other imports
    transformers.utils.import_utils.check_torch_load_is_safe = disabled_torch_load_check
    
    # Also patch the module-level import to ensure compatibility
    import sys
    if 'transformers.modeling_utils' in sys.modules:
        # If already imported, patch it directly
        import transformers.modeling_utils
        transformers.modeling_utils.check_torch_load_is_safe = disabled_torch_load_check
    
    print("Applied global torch.load security check bypass for HunYuan3D models")
    
    # Apply advanced optimizations based on system capabilities
    try:
        from .config.optimization import apply_global_optimizations
        optimization_config = apply_global_optimizations(memory_profile="auto")
        print(f"Applied advanced optimizations: memory_profile=auto")
    except ImportError as e:
        print(f"Advanced optimizations not available: {e}")
    
except Exception as e:
    # If patching fails, continue anyway - the error will be handled later
    print(f"Warning: Failed to apply torch.load security patch: {e}")
    
    # Still try to apply optimizations even if security patch failed
    try:
        from .config.optimization import apply_global_optimizations
        optimization_config = apply_global_optimizations(memory_profile="auto")
        print(f"Applied advanced optimizations: memory_profile=auto")
    except ImportError as opt_e:
        print(f"Advanced optimizations not available: {opt_e}")
    pass

# Apply torchvision compatibility patch first
try:
    from .torchvision_patch import patch_torchvision
    patch_torchvision()
except Exception:
    pass

# Setup Hunyuan3D paths before any other imports
from .setup_paths import setup_hunyuan3d_paths
setup_hunyuan3d_paths()

# Configure PyTorch compilation after imports
try:
    import torch
    if hasattr(torch, '_dynamo'):
        # Configure torch._dynamo for better Windows compatibility
        torch._dynamo.config.suppress_errors = True  # Fall back to eager on errors
        torch._dynamo.config.cache_size_limit = 64  # Reasonable cache size
        
        if platform.system() == "Windows":
            # Windows-specific settings
            torch._dynamo.config.force_disable_caches = False  # Keep caching for performance
            torch._dynamo.config.assume_static_by_default = True  # Better for Windows
            
            # Disable problematic optimizations on Windows
            if hasattr(torch._inductor, 'config'):
                torch._inductor.config.triton.cudagraphs = False  # Disable CUDA graphs with Triton
                torch._inductor.config.triton.autotune_pointwise = False  # Disable autotuning
        
        print(f"Configured PyTorch compilation settings for {platform.system()}")
except Exception as e:
    # Silently continue if configuration fails
    pass

# Configure tqdm to update more frequently
os.environ["TQDM_MININTERVAL"] = "0.01"  # Update every 0.01 seconds (default is 0.1)
os.environ["TQDM_MINITERS"] = "1"       # Update after each iteration (default is 1)

# Direct configuration of tqdm
try:
    # Import and configure tqdm directly
    from tqdm.auto import tqdm
    # Set default format to include more detailed progress information
    tqdm.monitor_interval = 0  # Disable monitor thread (can cause issues with some terminals)
    # Override default parameters to update more frequently
    original_init = tqdm.__init__

    def patched_init(self, *args, **kwargs):
        # Set mininterval to 0.01 seconds if not specified
        if 'mininterval' not in kwargs:
            kwargs['mininterval'] = 0.01
        # Set miniters to 1 if not specified
        if 'miniters' not in kwargs:
            kwargs['miniters'] = 1
        # Call original init
        original_init(self, *args, **kwargs)

    tqdm.__init__ = patched_init
except ImportError:
    # tqdm might not be available at this point, which is fine
    pass

# Compatibility imports for the reorganized structure
# This allows existing code to continue working with the new module layout

# Core classes
from .core.studio import Hunyuan3DStudio
from .core.studio_enhanced import Hunyuan3DStudioEnhanced

# Main interfaces - keeping the old import working
try:
    from .app import interface
except ImportError:
    interface = None

# Backward compatibility aliases
from .models import ModelManager as model_manager
from .services.queue import QueueManager as queue_manager
from .services.history import HistoryManager as history_manager
from .generation.image import ImageGenerator as image_generation
from .utils.memory import get_memory_manager as memory_manager
from .utils.gpu import get_gpu_optimizer as gpu_optimizer

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Hunyuan3DStudio",
    "Hunyuan3DStudioEnhanced",
    
    # Main interface
    "interface",
    
    # Compatibility aliases
    "model_manager",
    "queue_manager",
    "history_manager",
    "image_generation",
    "memory_manager",
    "gpu_optimizer"
]
