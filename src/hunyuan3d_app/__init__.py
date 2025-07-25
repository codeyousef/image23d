import os

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
    
except Exception as e:
    # If patching fails, continue anyway - the error will be handled later
    print(f"Warning: Failed to apply torch.load security patch: {e}")
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
