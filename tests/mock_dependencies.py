"""Mock missing dependencies for testing."""

import sys
from unittest.mock import MagicMock

# Mock torch
if 'torch' not in sys.modules:
    torch_mock = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=False)
    torch_mock.cuda.empty_cache = MagicMock()
    torch_mock.cuda.memory_allocated = MagicMock(return_value=0)
    torch_mock.cuda.mem_get_info = MagicMock(return_value=(0, 0))
    torch_mock.cuda.device_count = MagicMock(return_value=0)
    torch_mock.manual_seed = MagicMock()
    torch_mock.cuda.manual_seed_all = MagicMock()
    torch_mock.device = MagicMock(return_value='cpu')
    torch_mock.float32 = 'float32'
    torch_mock.float16 = 'float16'
    sys.modules['torch'] = torch_mock

# Mock transformers
if 'transformers' not in sys.modules:
    transformers_mock = MagicMock()
    sys.modules['transformers'] = transformers_mock
    sys.modules['transformers.pipeline'] = MagicMock()

# Mock diffusers
if 'diffusers' not in sys.modules:
    diffusers_mock = MagicMock()
    sys.modules['diffusers'] = diffusers_mock

# Mock trimesh
if 'trimesh' not in sys.modules:
    trimesh_mock = MagicMock()
    trimesh_mock.creation = MagicMock()
    trimesh_mock.creation.box = MagicMock()
    trimesh_mock.Trimesh = MagicMock()
    sys.modules['trimesh'] = trimesh_mock

# Mock cv2
if 'cv2' not in sys.modules:
    cv2_mock = MagicMock()
    cv2_mock.VideoWriter = MagicMock()
    sys.modules['cv2'] = cv2_mock

# Mock keyring
if 'keyring' not in sys.modules:
    keyring_mock = MagicMock()
    keyring_mock.set_password = MagicMock()
    keyring_mock.get_password = MagicMock(return_value=None)
    keyring_mock.delete_password = MagicMock()
    sys.modules['keyring'] = keyring_mock

# Mock cryptography
if 'cryptography' not in sys.modules:
    cryptography_mock = MagicMock()
    sys.modules['cryptography'] = cryptography_mock
    sys.modules['cryptography.fernet'] = MagicMock()

# Mock huggingface_hub
if 'huggingface_hub' not in sys.modules:
    hf_mock = MagicMock()
    hf_mock.snapshot_download = MagicMock(return_value='/mock/path')
    hf_mock.hf_hub_download = MagicMock(return_value='/mock/file')
    hf_mock.HfFolder = MagicMock()
    hf_mock.HfFolder.get_token = MagicMock(return_value=None)
    hf_mock.login = MagicMock()
    
    # Mock utils module
    hf_utils_mock = MagicMock()
    hf_utils_mock.RepositoryNotFoundError = type('RepositoryNotFoundError', (Exception,), {})
    hf_utils_mock.HfHubHTTPError = type('HfHubHTTPError', (Exception,), {})
    hf_mock.utils = hf_utils_mock
    
    sys.modules['huggingface_hub'] = hf_mock
    sys.modules['huggingface_hub.utils'] = hf_utils_mock

# Mock insightface
if 'insightface' not in sys.modules:
    insightface_mock = MagicMock()
    sys.modules['insightface'] = insightface_mock
    sys.modules['insightface.app'] = MagicMock()
    sys.modules['insightface.model_zoo'] = MagicMock()

# Mock gradio
if 'gradio' not in sys.modules:
    gradio_mock = MagicMock()
    gradio_mock.Progress = MagicMock
    sys.modules['gradio'] = gradio_mock
    sys.modules['gr'] = gradio_mock

# Mock gradio_client before it's imported
if 'gradio_client' not in sys.modules:
    gradio_client_mock = MagicMock()
    gradio_client_mock.utils = MagicMock()
    sys.modules['gradio_client'] = gradio_client_mock
    sys.modules['gradio_client.utils'] = gradio_client_mock.utils

# Mock aiohttp
if 'aiohttp' not in sys.modules:
    aiohttp_mock = MagicMock()
    aiohttp_mock.ClientSession = MagicMock
    sys.modules['aiohttp'] = aiohttp_mock

# Mock websockets
if 'websockets' not in sys.modules:
    websockets_mock = MagicMock()
    sys.modules['websockets'] = websockets_mock
    sys.modules['websockets.server'] = MagicMock()

# Mock tqdm
if 'tqdm' not in sys.modules:
    tqdm_mock = MagicMock()
    tqdm_mock.tqdm = MagicMock
    sys.modules['tqdm'] = tqdm_mock