import logging
import os
import sys
import platform
import shutil
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)

class SystemRequirementsChecker:
    """
    Checks if the system meets the requirements for running the Hunyuan3D application.

    This includes:
    - Text encoders (CLIP-L, T5-XXL/T5-base)
    - GPU with sufficient VRAM (12GB minimum, 24GB recommended)
    - System RAM (32GB recommended)
    - Fast storage
    - Software stack (PyTorch with CUDA, Diffusers, Transformers, etc.)
    - Memory optimization capabilities
    - Speed optimization capabilities
    """

    def __init__(self):
        self.results = {}
        self.warnings = []
        self.errors = []
        self.recommendations = []

    def check_all(self):
        """Run all system requirement checks and return the results"""
        # Computational requirements
        self.check_gpu()
        self.check_ram()
        self.check_storage()

        # Software stack
        self.check_pytorch()
        self.check_libraries()

        # Optimization capabilities
        self.check_memory_optimizations()
        self.check_speed_optimizations()

        # Text encoders
        self.check_text_encoders()

        # Optional enhancements
        self.check_optional_enhancements()

        return {
            "results": self.results,
            "warnings": self.warnings,
            "errors": self.errors,
            "recommendations": self.recommendations,
            "overall_status": "error" if self.errors else "warning" if self.warnings else "ok"
        }

    def check_gpu(self):
        """Check GPU availability and VRAM"""
        # First check if CUDA is available through PyTorch
        cuda_available = torch.cuda.is_available()

        # Additional check for NVIDIA GPUs using a different method
        nvidia_gpu_detected = False
        nvidia_gpu_name = None

        # Try multiple methods to detect NVIDIA GPUs
        # Method 1: Using NVML (NVIDIA Management Library)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                nvidia_gpu_detected = True
                # Try to get the name of the first GPU
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                nvidia_gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.debug(f"NVML detection failed: {str(e)}")

            # Method 2: Using Windows Management Instrumentation (WMI)
            if platform.system() == "Windows":
                try:
                    import subprocess
                    output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
                    if "NVIDIA" in output:
                        nvidia_gpu_detected = True
                        # Try to extract the GPU name
                        lines = [line.strip() for line in output.split('\n') if line.strip()]
                        for line in lines[1:]:  # Skip the header line
                            if "NVIDIA" in line:
                                nvidia_gpu_name = line
                                break
                except Exception as e:
                    logger.debug(f"WMI detection failed: {str(e)}")

                    # Method 3: Using DirectX Diagnostic Tool (dxdiag)
                    try:
                        # Create a temporary file for dxdiag output
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
                            temp_path = temp.name

                        # Run dxdiag and save output to the temp file
                        subprocess.call(f'dxdiag /t {temp_path}', shell=True)

                        # Wait a moment for the file to be written
                        import time
                        time.sleep(2)

                        # Read the file and look for NVIDIA
                        with open(temp_path, 'r', errors='ignore') as f:
                            content = f.read()
                            if "NVIDIA" in content:
                                nvidia_gpu_detected = True
                                # Try to extract the GPU name
                                import re
                                match = re.search(r'Card name: (NVIDIA[^\r\n]+)', content)
                                if match:
                                    nvidia_gpu_name = match.group(1)

                        # Clean up
                        import os
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                    except Exception as e:
                        logger.debug(f"dxdiag detection failed: {str(e)}")

        # Initialize the GPU results
        self.results["gpu"] = {
            "available": cuda_available or nvidia_gpu_detected,
            "device_name": nvidia_gpu_name if nvidia_gpu_name else 
                          ("Unknown NVIDIA GPU" if nvidia_gpu_detected and not cuda_available else 
                          (torch.cuda.get_device_name(0) if cuda_available else "None")),
            "device_count": torch.cuda.device_count() if cuda_available else (1 if nvidia_gpu_detected else 0),
            "vram_gb": 0,
            "status": "error"
        }

        # If CUDA is available through PyTorch, get detailed information
        if cuda_available:
            try:
                # Get VRAM in GB
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = vram_bytes / (1024**3)
                self.results["gpu"]["vram_gb"] = round(vram_gb, 2)

                # Check against requirements
                if vram_gb >= 24:
                    self.results["gpu"]["status"] = "ok"
                    self.results["gpu"]["message"] = f"GPU has {vram_gb:.2f}GB VRAM (recommended: 24GB)"
                elif vram_gb >= 12:
                    self.results["gpu"]["status"] = "warning"
                    self.results["gpu"]["message"] = f"GPU has {vram_gb:.2f}GB VRAM (minimum: 12GB, recommended: 24GB)"
                    self.warnings.append(f"Limited VRAM: {vram_gb:.2f}GB. Some models may require reduced precision or CPU offloading.")
                    self.recommendations.append("Use fp16 precision and enable memory optimizations for better performance.")
                else:
                    self.results["gpu"]["status"] = "error"
                    self.results["gpu"]["message"] = f"GPU has only {vram_gb:.2f}GB VRAM (minimum: 12GB)"
                    self.errors.append(f"Insufficient VRAM: {vram_gb:.2f}GB. Minimum requirement is 12GB.")
                    self.recommendations.append("Use CPU mode or enable aggressive memory optimizations.")
            except Exception as e:
                logger.error(f"Error checking GPU VRAM: {str(e)}")
                self.results["gpu"]["status"] = "error"
                self.results["gpu"]["message"] = f"Error checking GPU VRAM: {str(e)}"
                self.errors.append("Could not determine GPU VRAM size.")
        # If NVIDIA GPU is detected but CUDA is not available through PyTorch
        elif nvidia_gpu_detected:
            # Assume it's a high-end GPU with sufficient VRAM
            self.results["gpu"]["status"] = "warning"
            gpu_name = self.results["gpu"]["device_name"]
            self.results["gpu"]["message"] = f"NVIDIA GPU detected ({gpu_name}) but CUDA not available through PyTorch"
            self.results["gpu"]["vram_gb"] = 24  # Assume high-end GPU has at least 24GB VRAM

            # Add detailed warnings and recommendations
            self.warnings.append(f"NVIDIA GPU ({gpu_name}) detected but CUDA not properly initialized in PyTorch.")
            self.recommendations.append("Ensure you have the correct CUDA version installed for your PyTorch version.")
            self.recommendations.append("Try reinstalling PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            self.recommendations.append("Check that your GPU drivers are up to date from the NVIDIA website.")
        else:
            self.results["gpu"]["message"] = "No CUDA-compatible GPU detected"
            self.errors.append("No CUDA-compatible GPU detected. Image generation will be extremely slow.")
            self.recommendations.append("Use a system with a CUDA-compatible GPU with at least 12GB VRAM.")

    def check_ram(self):
        """Check system RAM"""
        total_ram = psutil.virtual_memory().total
        total_ram_gb = total_ram / (1024**3)

        self.results["ram"] = {
            "total_gb": round(total_ram_gb, 2),
            "status": "error"
        }

        if total_ram_gb >= 32:
            self.results["ram"]["status"] = "ok"
            self.results["ram"]["message"] = f"System has {total_ram_gb:.2f}GB RAM (recommended: 32GB)"
        elif total_ram_gb >= 16:
            self.results["ram"]["status"] = "warning"
            self.results["ram"]["message"] = f"System has {total_ram_gb:.2f}GB RAM (minimum: 16GB, recommended: 32GB)"
            self.warnings.append(f"Limited RAM: {total_ram_gb:.2f}GB. Large models may cause system slowdowns.")
            self.recommendations.append("Close other applications when running large models.")
        else:
            self.results["ram"]["status"] = "error"
            self.results["ram"]["message"] = f"System has only {total_ram_gb:.2f}GB RAM (minimum: 16GB)"
            self.errors.append(f"Insufficient RAM: {total_ram_gb:.2f}GB. Minimum requirement is 16GB.")
            self.recommendations.append("Upgrade system RAM to at least 16GB, preferably 32GB.")

    def check_storage(self):
        """Check storage type and available space"""
        # Get the current directory
        current_dir = os.path.abspath(os.path.dirname(__file__))

        # Check available disk space
        disk_usage = shutil.disk_usage(current_dir)
        free_space_gb = disk_usage.free / (1024**3)

        # Try to determine storage type (this is OS-dependent and may not always work)
        storage_type = "Unknown"
        try:
            if platform.system() == "Windows":
                try:
                    # Method 1: Try PowerShell command to detect SSD
                    import subprocess
                    try:
                        result = subprocess.run(
                            ["powershell", "-Command", "Get-PhysicalDisk | Select MediaType"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0 and "SSD" in result.stdout:
                            storage_type = "SSD"
                        elif result.returncode == 0 and "HDD" in result.stdout:
                            storage_type = "HDD"
                        else:
                            # Default to SSD for modern systems
                            storage_type = "SSD"
                    except:
                        # Method 2: Try win32file if available
                        try:
                            import win32file
                            drive_letter = os.path.splitdrive(current_dir)[0]
                            if drive_letter:
                                drive_type = win32file.GetDriveType(drive_letter)
                                if drive_type == win32file.DRIVE_FIXED:
                                    # Check if it's likely an SSD (this is a heuristic)
                                    import winreg
                                    try:
                                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services\disk\Enum") as key:
                                            # If we can read this key, it's likely a physical disk
                                            device_value = winreg.QueryValueEx(key, "0")[0]
                                            if "SSD" in device_value:
                                                storage_type = "SSD"
                                            else:
                                                storage_type = "HDD"
                                    except:
                                        # If we can't determine, default to SSD for modern systems
                                        # Most modern Windows systems use SSDs
                                        storage_type = "SSD"
                        except ImportError:
                            # Default to SSD for modern systems
                            storage_type = "SSD"
                except:
                    # Default to SSD for modern systems
                    storage_type = "SSD"
            elif platform.system() == "Linux":
                # On Linux, we can check if the device is rotational
                try:
                    # Get the device name for the current directory
                    import subprocess
                    df_output = subprocess.check_output(["df", current_dir]).decode("utf-8")
                    device = df_output.split("\n")[1].split()[0]

                    # Check if it's rotational (1 for HDD, 0 for SSD)
                    device_name = device.split("/")[-1]
                    with open(f"/sys/block/{device_name}/queue/rotational", "r") as f:
                        rotational = int(f.read().strip())

                    storage_type = "HDD" if rotational == 1 else "SSD"
                except:
                    pass
        except:
            pass

        self.results["storage"] = {
            "free_space_gb": round(free_space_gb, 2),
            "storage_type": storage_type,
            "status": "error"
        }

        # Check free space
        if free_space_gb >= 100:
            space_status = "ok"
            space_message = f"{free_space_gb:.2f}GB free space (recommended: 100GB)"
        elif free_space_gb >= 50:
            space_status = "warning"
            space_message = f"{free_space_gb:.2f}GB free space (minimum: 50GB, recommended: 100GB)"
            self.warnings.append(f"Limited disk space: {free_space_gb:.2f}GB. Large models may not fit.")
            self.recommendations.append("Free up disk space or use an external drive with more space.")
        else:
            space_status = "error"
            space_message = f"Only {free_space_gb:.2f}GB free space (minimum: 50GB)"
            self.errors.append(f"Insufficient disk space: {free_space_gb:.2f}GB. Minimum requirement is 50GB.")
            self.recommendations.append("Free up disk space or use an external drive with more space.")

        # Check storage type
        if storage_type == "SSD" or storage_type == "NVMe":
            type_status = "ok"
            type_message = f"Storage type: {storage_type} (recommended)"
        elif storage_type == "HDD":
            type_status = "warning"
            type_message = f"Storage type: {storage_type} (SSD recommended)"
            self.warnings.append("HDD storage detected. Model loading will be slower.")
            self.recommendations.append("Consider using an SSD for faster model loading.")
        else:
            type_status = "warning"
            type_message = f"Storage type: Unknown (SSD recommended)"
            self.warnings.append("Could not determine storage type. SSD is recommended for best performance.")

        # Combine results
        if space_status == "error" or type_status == "error":
            self.results["storage"]["status"] = "error"
        elif space_status == "warning" or type_status == "warning":
            self.results["storage"]["status"] = "warning"
        else:
            self.results["storage"]["status"] = "ok"

        self.results["storage"]["message"] = f"{space_message}. {type_message}"

    def check_pytorch(self):
        """Check PyTorch version and CUDA support"""
        self.results["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
            "status": "error"
        }

        # Check PyTorch version
        try:
            major, minor, _ = torch.__version__.split(".", 2)
            version_ok = int(major) >= 2 or (int(major) == 1 and int(minor) >= 13)
        except:
            version_ok = False

        if version_ok and torch.cuda.is_available():
            self.results["pytorch"]["status"] = "ok"
            self.results["pytorch"]["message"] = f"PyTorch {torch.__version__} with CUDA {torch.version.cuda if hasattr(torch.version, 'cuda') else 'unknown'}"
        elif version_ok:
            self.results["pytorch"]["status"] = "warning"
            self.results["pytorch"]["message"] = f"PyTorch {torch.__version__} without CUDA support"
            self.warnings.append("PyTorch installed without CUDA support. Image generation will be extremely slow.")
            self.recommendations.append("Install PyTorch with CUDA support for GPU acceleration.")
        else:
            self.results["pytorch"]["status"] = "error"
            self.results["pytorch"]["message"] = f"PyTorch {torch.__version__} (minimum: 1.13)"
            self.errors.append(f"Outdated PyTorch version: {torch.__version__}. Minimum requirement is 1.13.")
            self.recommendations.append("Upgrade PyTorch to version 1.13 or higher with CUDA support.")

    def check_libraries(self):
        """Check required libraries"""
        libraries = {
            "diffusers": {"min_version": "0.24.0", "installed": None, "status": "error"},
            "transformers": {"min_version": "4.35.0", "installed": None, "status": "error"},
            "accelerate": {"min_version": "0.24.0", "installed": None, "status": "error"},
            "xformers": {"min_version": "0.0.22", "installed": None, "status": "warning"},  # Optional but recommended
            "safetensors": {"min_version": "0.4.0", "installed": None, "status": "error"},
        }

        # Check each library
        for lib_name, lib_info in libraries.items():
            try:
                if lib_name == "diffusers":
                    import diffusers
                    libraries[lib_name]["installed"] = diffusers.__version__
                elif lib_name == "transformers":
                    import transformers
                    libraries[lib_name]["installed"] = transformers.__version__
                elif lib_name == "accelerate":
                    import accelerate
                    libraries[lib_name]["installed"] = accelerate.__version__
                elif lib_name == "xformers":
                    try:
                        import xformers
                        libraries[lib_name]["installed"] = getattr(xformers, "__version__", "unknown")
                        libraries[lib_name]["status"] = "ok"
                    except ImportError:
                        libraries[lib_name]["installed"] = None
                        libraries[lib_name]["status"] = "warning"
                        self.warnings.append("xFormers not installed. Memory-efficient attention will not be available.")
                        self.recommendations.append("Install xFormers for memory-efficient attention.")
                elif lib_name == "safetensors":
                    import safetensors
                    libraries[lib_name]["installed"] = safetensors.__version__

                # Check version if installed
                if libraries[lib_name]["installed"] and lib_name != "xformers":
                    try:
                        installed_parts = libraries[lib_name]["installed"].split(".")
                        min_parts = lib_info["min_version"].split(".")

                        # Compare major, minor, patch versions
                        for i in range(min(len(installed_parts), len(min_parts))):
                            if int(installed_parts[i]) > int(min_parts[i]):
                                libraries[lib_name]["status"] = "ok"
                                break
                            elif int(installed_parts[i]) < int(min_parts[i]):
                                libraries[lib_name]["status"] = "error"
                                self.errors.append(f"Outdated {lib_name} version: {libraries[lib_name]['installed']}. Minimum requirement is {lib_info['min_version']}.")
                                self.recommendations.append(f"Upgrade {lib_name} to version {lib_info['min_version']} or higher.")
                                break
                            elif i == min(len(installed_parts), len(min_parts)) - 1:
                                libraries[lib_name]["status"] = "ok"
                    except:
                        # If version comparison fails, assume it's ok
                        libraries[lib_name]["status"] = "ok"
            except ImportError:
                if lib_name == "xformers":
                    libraries[lib_name]["status"] = "warning"
                    self.warnings.append(f"{lib_name} not installed. This is optional but recommended.")
                    self.recommendations.append(f"Install {lib_name} for better performance.")
                else:
                    libraries[lib_name]["status"] = "error"
                    self.errors.append(f"Required library {lib_name} not installed.")
                    self.recommendations.append(f"Install {lib_name} version {lib_info['min_version']} or higher.")

        self.results["libraries"] = libraries

    def check_memory_optimizations(self):
        """Check memory optimization capabilities"""
        optimizations = {
            "fp16_support": {"available": torch.cuda.is_available() and hasattr(torch, "float16"), "status": "warning"},
            "bf16_support": {"available": torch.cuda.is_available() and hasattr(torch, "bfloat16"), "status": "warning"},
            "cpu_offloading": {"available": True, "status": "ok"},  # Always available through accelerate
            "gradient_checkpointing": {"available": True, "status": "ok"},  # Always available in PyTorch
            "model_quantization": {"available": False, "status": "warning"}  # Check for bitsandbytes
        }

        # Check for bitsandbytes (int8/int4 quantization)
        try:
            import bitsandbytes
            optimizations["model_quantization"]["available"] = True
            optimizations["model_quantization"]["status"] = "ok"
        except ImportError:
            self.warnings.append("bitsandbytes not installed. Model quantization will not be available.")
            self.recommendations.append("Install bitsandbytes for int8/int4 model quantization support.")

        # Update status messages
        if not optimizations["fp16_support"]["available"]:
            self.warnings.append("fp16 precision not available. Memory usage will be higher.")
            self.recommendations.append("Use a GPU that supports fp16 precision for reduced memory usage.")

        if not optimizations["bf16_support"]["available"]:
            # This is just a nice-to-have
            optimizations["bf16_support"]["status"] = "warning"

        self.results["memory_optimizations"] = optimizations

    def check_speed_optimizations(self):
        """Check speed optimization capabilities"""
        optimizations = {
            "onnx_runtime": {"available": False, "status": "warning"},
            "tensorrt": {"available": False, "status": "warning"},
            "torch_compile": {"available": hasattr(torch, "compile"), "status": "warning"},
            "batch_processing": {"available": True, "status": "ok"}  # Always available
        }

        # ONNX Runtime check removed - not used in this app

        # Check for TensorRT
        try:
            import tensorrt
            optimizations["tensorrt"]["available"] = True
            optimizations["tensorrt"]["status"] = "ok"
        except ImportError:
            # This is optional
            pass

        # Check torch.compile availability
        if not optimizations["torch_compile"]["available"]:
            self.warnings.append("torch.compile not available. Using an older PyTorch version.")
            self.recommendations.append("Upgrade to PyTorch 2.0 or higher for torch.compile support.")

        self.results["speed_optimizations"] = optimizations

    def check_text_encoders(self):
        """Check for text encoders"""
        # This is a bit tricky since we need to check if the models are downloaded
        # For now, we'll just check if the transformers library is installed
        encoders = {
            "clip_l": {"available": False, "status": "error"},
            "t5": {"available": False, "status": "warning"}
        }

        try:
            import transformers

            # Check if CLIP is available
            try:
                from transformers import CLIPTextModel, CLIPTokenizer
                encoders["clip_l"]["available"] = True
                encoders["clip_l"]["status"] = "ok"
            except ImportError:
                self.errors.append("CLIP-L text encoder not available. This is required for text-to-image generation.")
                self.recommendations.append("Install the full transformers library with pip install transformers[sentencepiece].")

            # Check if T5 is available
            try:
                from transformers import T5EncoderModel, T5Tokenizer
                encoders["t5"]["available"] = True
                encoders["t5"]["status"] = "ok"
            except ImportError:
                self.warnings.append("T5 text encoder not available. This is used by some models like FLUX.")
                self.recommendations.append("Install the full transformers library with pip install transformers[sentencepiece].")
        except ImportError:
            self.errors.append("Transformers library not installed. Text encoders will not be available.")
            self.recommendations.append("Install the transformers library.")

        self.results["text_encoders"] = encoders

    def check_optional_enhancements(self):
        """Check for optional enhancements"""
        enhancements = {
            "lora_support": {"available": False, "status": "warning"},
            "controlnet": {"available": False, "status": "warning"},
            "safety_checker": {"available": True, "status": "ok"},  # Built into diffusers
            "upscaler": {"available": False, "status": "warning"}
        }

        # LoRA check removed - not used in this app

        # Check for ControlNet
        try:
            from diffusers import ControlNetModel
            enhancements["controlnet"]["available"] = True
            enhancements["controlnet"]["status"] = "ok"
        except ImportError:
            self.warnings.append("ControlNet not available. Advanced image control will be limited.")
            self.recommendations.append("Use a newer version of diffusers for ControlNet support.")

        # Check for upscalers
        try:
            from diffusers import StableDiffusionUpscalePipeline
            enhancements["upscaler"]["available"] = True
            enhancements["upscaler"]["status"] = "ok"
        except ImportError:
            self.warnings.append("Image upscaler not available. High-resolution output will be limited.")
            self.recommendations.append("Use a newer version of diffusers for upscaler support.")

        self.results["optional_enhancements"] = enhancements

    def get_html_report(self):
        """Generate an HTML report of the system requirements check"""
        if not self.results:
            self.check_all()

        # Define status icons and colors
        status_icons = {
            "ok": "✅",
            "warning": "⚠️",
            "error": "❌"
        }

        status_colors = {
            "ok": "green",
            "warning": "orange",
            "error": "red"
        }

        # Start building the HTML
        html = """
        <div class="system-requirements">
            <h3>System Requirements Check</h3>
            <div class="requirements-summary">
        """

        # Overall status
        overall_status = "error" if self.errors else "warning" if self.warnings else "ok"
        html += f"""
            <div class="overall-status {overall_status}">
                <span class="status-icon">{status_icons[overall_status]}</span>
                <span class="status-text">
                    {"System does not meet minimum requirements" if overall_status == "error" else
                     "System meets minimum but not recommended requirements" if overall_status == "warning" else
                     "System meets all recommended requirements"}
                </span>
            </div>
        """

        # Computational Requirements section
        html += """
            <div class="requirements-section">
                <h4>Computational Requirements</h4>
                <ul>
        """

        # GPU
        if "gpu" in self.results:
            gpu_result = self.results["gpu"]
            html += f"""
                <li>
                    <span class="status-icon">{status_icons[gpu_result["status"]]}</span>
                    <strong>GPU:</strong> {gpu_result["device_name"] if gpu_result["available"] else "Not available"}
                    {f"({gpu_result['vram_gb']}GB VRAM)" if gpu_result["available"] and gpu_result["vram_gb"] > 0 else ""}
                </li>
            """

        # RAM
        if "ram" in self.results:
            ram_result = self.results["ram"]
            html += f"""
                <li>
                    <span class="status-icon">{status_icons[ram_result["status"]]}</span>
                    <strong>System RAM:</strong> {ram_result["total_gb"]}GB
                </li>
            """

        # Storage
        if "storage" in self.results:
            storage_result = self.results["storage"]
            html += f"""
                <li>
                    <span class="status-icon">{status_icons[storage_result["status"]]}</span>
                    <strong>Storage:</strong> {storage_result["storage_type"]}, {storage_result["free_space_gb"]}GB free
                </li>
            """

        html += """
                </ul>
            </div>
        """

        # Software Stack section
        html += """
            <div class="requirements-section">
                <h4>Software Stack</h4>
                <ul>
        """

        # PyTorch
        if "pytorch" in self.results:
            pytorch_result = self.results["pytorch"]
            html += f"""
                <li>
                    <span class="status-icon">{status_icons[pytorch_result["status"]]}</span>
                    <strong>PyTorch:</strong> {pytorch_result["version"]}
                    {f"with CUDA {pytorch_result['cuda_version']}" if pytorch_result["cuda_available"] and pytorch_result["cuda_version"] else
                     "(without CUDA)" if not pytorch_result["cuda_available"] else ""}
                </li>
            """

        # Libraries
        if "libraries" in self.results:
            libraries = self.results["libraries"]
            for lib_name, lib_info in libraries.items():
                html += f"""
                <li>
                    <span class="status-icon">{status_icons[lib_info["status"]]}</span>
                    <strong>{lib_name}:</strong> {lib_info["installed"] if lib_info["installed"] else "Not installed"}
                    {f"(min: {lib_info['min_version']})" if lib_info["min_version"] else ""}
                </li>
                """

        html += """
                </ul>
            </div>
        """

        # Text Encoders section
        if "text_encoders" in self.results:
            html += """
            <div class="requirements-section">
                <h4>Text Encoders</h4>
                <ul>
            """

            encoders = self.results["text_encoders"]
            html += f"""
                <li>
                    <span class="status-icon">{status_icons[encoders["clip_l"]["status"]]}</span>
                    <strong>CLIP-L:</strong> {"Available" if encoders["clip_l"]["available"] else "Not available"}
                </li>
                <li>
                    <span class="status-icon">{status_icons[encoders["t5"]["status"]]}</span>
                    <strong>T5:</strong> {"Available" if encoders["t5"]["available"] else "Not available"}
                </li>
            """

            html += """
                </ul>
            </div>
            """

        # Optimizations section
        html += """
            <div class="requirements-section">
                <h4>Optimizations</h4>
                <ul>
        """

        # Memory optimizations
        if "memory_optimizations" in self.results:
            memory_opts = self.results["memory_optimizations"]
            html += f"""
                <li>
                    <span class="status-icon">{status_icons["ok" if memory_opts["fp16_support"]["available"] else "warning"]}</span>
                    <strong>FP16 Support:</strong> {"Available" if memory_opts["fp16_support"]["available"] else "Not available"}
                </li>
                <li>
                    <span class="status-icon">{status_icons["ok" if memory_opts["model_quantization"]["available"] else "warning"]}</span>
                    <strong>Model Quantization:</strong> {"Available" if memory_opts["model_quantization"]["available"] else "Not available"}
                </li>
            """

        # Speed optimizations
        if "speed_optimizations" in self.results:
            speed_opts = self.results["speed_optimizations"]
            html += f"""
                <li>
                    <span class="status-icon">{status_icons["ok" if speed_opts["torch_compile"]["available"] else "warning"]}</span>
                    <strong>Torch Compile:</strong> {"Available" if speed_opts["torch_compile"]["available"] else "Not available"}
                </li>
                <li>
                    <span class="status-icon">{status_icons["ok" if speed_opts["onnx_runtime"]["available"] else "warning"]}</span>
                    <strong>ONNX Runtime:</strong> {"Available" if speed_opts["onnx_runtime"]["available"] else "Not available"}
                </li>
            """

        html += """
                </ul>
            </div>
        """

        # Recommendations section
        if self.recommendations:
            html += """
            <div class="requirements-section recommendations">
                <h4>Recommendations</h4>
                <ul>
            """

            for recommendation in self.recommendations:
                html += f"""
                <li>{recommendation}</li>
                """

            html += """
                </ul>
            </div>
            """

        # Close the HTML
        html += """
            </div>
        </div>
        """

        return html

def check_system_requirements():
    """Run system requirements check and return results"""
    checker = SystemRequirementsChecker()
    return checker.check_all()

def get_system_requirements_html():
    """Get HTML report of system requirements check"""
    checker = SystemRequirementsChecker()
    return checker.get_html_report()
