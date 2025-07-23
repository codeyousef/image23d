"""Application-wide constants."""

# --- Supported File Formats ---
SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
SUPPORTED_3D_FORMATS = {'.glb', '.obj', '.ply', '.stl', '.fbx', '.usdz'}

# --- Default Values ---
DEFAULT_IMAGE_STEPS = 20
DEFAULT_IMAGE_GUIDANCE = 7.5
DEFAULT_3D_VIEWS = 6
DEFAULT_MESH_RESOLUTION = 512
DEFAULT_TEXTURE_RESOLUTION = 2048

# --- Memory Limits ---
MIN_VRAM_GB = 4
RECOMMENDED_VRAM_GB = 8
OPTIMAL_VRAM_GB = 16

# --- Queue Settings ---
MAX_QUEUE_SIZE = 100
DEFAULT_BATCH_SIZE = 4
MAX_BATCH_SIZE = 16

# --- WebSocket Settings ---
WEBSOCKET_TIMEOUT = 30
WEBSOCKET_PING_INTERVAL = 10

# --- API Limits ---
MAX_API_RETRIES = 3
API_TIMEOUT = 300  # 5 minutes

# --- File Size Limits ---
MAX_IMAGE_SIZE_MB = 50
MAX_VIDEO_SIZE_MB = 500
MAX_3D_SIZE_MB = 200

# --- Generation Limits ---
MAX_IMAGE_WIDTH = 2048
MAX_IMAGE_HEIGHT = 2048
MIN_IMAGE_SIZE = 256

# --- Progress Update Intervals ---
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds