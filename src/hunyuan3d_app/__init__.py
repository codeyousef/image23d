import os

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

from .app import interface

__version__ = "0.1.0"
