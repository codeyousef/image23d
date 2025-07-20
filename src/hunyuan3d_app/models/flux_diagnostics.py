"""Diagnostic utilities for troubleshooting FLUX pipeline issues.

This module provides comprehensive debugging and troubleshooting tools
for the FLUX pipeline implementation.
"""

import torch
import logging
import sys
import os
import platform
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import psutil
import transformers

# Optional imports
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
import diffusers
from packaging import version

logger = logging.getLogger(__name__)


class FluxDiagnostics:
    """Comprehensive diagnostics for FLUX pipeline troubleshooting."""
    
    @staticmethod
    def run_full_diagnostic() -> Dict[str, Any]:
        """Run complete diagnostic check and return results."""
        
        logger.info("Running FLUX diagnostics...")
        logger.info("=" * 60)
        
        results = {
            "system": FluxDiagnostics.check_system(),
            "cuda": FluxDiagnostics.check_cuda(),
            "dependencies": FluxDiagnostics.check_dependencies(),
            "memory": FluxDiagnostics.check_memory(),
            "models": FluxDiagnostics.check_model_availability(),
            "common_issues": FluxDiagnostics.check_common_issues(),
            "recommendations": []
        }
        
        # Generate recommendations based on findings
        results["recommendations"] = FluxDiagnostics.generate_recommendations(results)
        
        # Print summary
        FluxDiagnostics.print_diagnostic_summary(results)
        
        return results
    
    @staticmethod
    def check_system() -> Dict[str, Any]:
        """Check system information."""
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": psutil.virtual_memory().total / 1024**3
        }
    
    @staticmethod
    def check_cuda() -> Dict[str, Any]:
        """Check CUDA availability and configuration."""
        cuda_info = {
            "available": torch.cuda.is_available(),
            "version": None,
            "device_count": 0,
            "devices": [],
            "cudnn_enabled": False,
            "current_device": None
        }
        
        if torch.cuda.is_available():
            cuda_info["version"] = torch.version.cuda
            cuda_info["device_count"] = torch.cuda.device_count()
            cuda_info["cudnn_enabled"] = torch.backends.cudnn.enabled
            cuda_info["current_device"] = torch.cuda.current_device()
            
            # Get detailed GPU info
            for i in range(cuda_info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                cuda_info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / 1024**3,
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count
                })
        
        return cuda_info
    
    @staticmethod
    def check_dependencies() -> Dict[str, Any]:
        """Check required dependencies and versions."""
        deps = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "diffusers": diffusers.__version__,
            "accelerate": None,
            "xformers": None,
            "safetensors": None,
            "optimum": None
        }
        
        # Check optional dependencies
        try:
            import accelerate
            deps["accelerate"] = accelerate.__version__
        except ImportError:
            pass
        
        try:
            import xformers
            deps["xformers"] = xformers.__version__
        except ImportError:
            pass
        
        try:
            import safetensors
            deps["safetensors"] = safetensors.__version__
        except ImportError:
            pass
        
        try:
            import optimum
            deps["optimum"] = optimum.__version__
        except ImportError:
            pass
        
        # Check compatibility
        compatibility = FluxDiagnostics._check_version_compatibility(deps)
        
        return {
            "versions": deps,
            "compatibility": compatibility
        }
    
    @staticmethod
    def _check_version_compatibility(deps: Dict[str, Optional[str]]) -> Dict[str, bool]:
        """Check if dependency versions are compatible."""
        compat = {
            "torch_cuda_match": True,
            "diffusers_transformers_match": True,
            "all_compatible": True
        }
        
        # Check if PyTorch CUDA version matches system CUDA
        if torch.cuda.is_available():
            torch_cuda = torch.version.cuda
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if "CUDA Version" in result.stdout:
                    # Extract CUDA version from nvidia-smi
                    import re
                    match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
                    if match:
                        system_cuda = match.group(1)
                        # Compare major versions
                        if torch_cuda and system_cuda:
                            torch_major = int(torch_cuda.split('.')[0])
                            system_major = int(system_cuda.split('.')[0])
                            compat["torch_cuda_match"] = torch_major == system_major
            except:
                pass
        
        # Check diffusers/transformers compatibility
        if deps["diffusers"] and deps["transformers"]:
            # These should generally be compatible if installed together
            pass
        
        compat["all_compatible"] = all(compat.values())
        
        return compat
    
    @staticmethod
    def check_memory() -> Dict[str, Any]:
        """Check memory availability and usage."""
        memory_info = {
            "ram": {},
            "gpu": []
        }
        
        # RAM info
        ram = psutil.virtual_memory()
        memory_info["ram"] = {
            "total_gb": ram.total / 1024**3,
            "available_gb": ram.available / 1024**3,
            "used_gb": ram.used / 1024**3,
            "percent": ram.percent
        }
        
        # GPU memory info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                memory_info["gpu"].append({
                    "device": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_gb": total,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "free_gb": total - allocated
                })
        
        return memory_info
    
    @staticmethod
    def check_model_availability() -> Dict[str, Any]:
        """Check which FLUX models are available/accessible."""
        models = {
            "flux_dev": {
                "repo_id": "black-forest-labs/FLUX.1-dev",
                "accessible": False,
                "requires_auth": True
            },
            "flux_schnell": {
                "repo_id": "black-forest-labs/FLUX.1-schnell",
                "accessible": False,
                "requires_auth": True
            },
            "flux_gguf_q8": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "accessible": False,
                "requires_auth": False
            },
            "flux_gguf_q4": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "accessible": False,
                "requires_auth": False
            }
        }
        
        # Check Hugging Face token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        has_token = bool(hf_token)
        
        # Note: Actual accessibility check would require network calls
        # This is a simplified version
        for model_name, model_info in models.items():
            if not model_info["requires_auth"]:
                model_info["accessible"] = True
            elif has_token:
                model_info["accessible"] = True
        
        return {
            "models": models,
            "hf_token_available": has_token
        }
    
    @staticmethod
    def check_common_issues() -> List[Dict[str, Any]]:
        """Check for common FLUX issues."""
        issues = []
        
        # Issue 1: Insufficient VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_gb < 8:
                issues.append({
                    "severity": "high",
                    "issue": "Insufficient VRAM",
                    "details": f"Only {vram_gb:.1f}GB VRAM available. FLUX requires at least 8GB.",
                    "solution": "Use GGUF quantized models (Q4 or Q3) or CPU offloading."
                })
        
        # Issue 2: Missing xFormers
        try:
            import xformers
        except ImportError:
            issues.append({
                "severity": "medium",
                "issue": "xFormers not installed",
                "details": "xFormers can significantly reduce memory usage and improve speed.",
                "solution": "Install with: pip install xformers"
            })
        
        # Issue 3: Old PyTorch version
        torch_version = version.parse(torch.__version__.split('+')[0])
        if torch_version < version.parse("2.0.0"):
            issues.append({
                "severity": "high",
                "issue": "Outdated PyTorch",
                "details": f"PyTorch {torch.__version__} is too old for optimal FLUX performance.",
                "solution": "Upgrade PyTorch to 2.0 or newer."
            })
        
        # Issue 4: No CUDA available
        if not torch.cuda.is_available():
            issues.append({
                "severity": "high",
                "issue": "No CUDA GPU available",
                "details": "FLUX will run very slowly on CPU only.",
                "solution": "Use a system with NVIDIA GPU or cloud GPU instances."
            })
        
        # Issue 5: Missing accelerate
        try:
            import accelerate
        except ImportError:
            issues.append({
                "severity": "low",
                "issue": "Accelerate not installed",
                "details": "Accelerate helps with multi-GPU and mixed precision training.",
                "solution": "Install with: pip install accelerate"
            })
        
        return issues
    
    @staticmethod
    def generate_recommendations(results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []
        
        # Memory recommendations
        if results["memory"]["gpu"]:
            gpu_mem = results["memory"]["gpu"][0]["total_gb"]
            if gpu_mem < 8:
                recommendations.append(
                    "Use GGUF Q4 or Q3 models for your GPU memory size."
                )
            elif gpu_mem < 12:
                recommendations.append(
                    "Use GGUF Q6 model for optimal quality/performance balance."
                )
            elif gpu_mem < 16:
                recommendations.append(
                    "Use GGUF Q8 model or base model with CPU offloading."
                )
            else:
                recommendations.append(
                    "You have sufficient VRAM for full FLUX models."
                )
        
        # Performance recommendations
        if results["cuda"]["available"]:
            recommendations.append(
                "Enable torch.compile() for 30-50% speed improvement."
            )
            recommendations.append(
                "Use HyperFlux or FluxTurbo LoRAs for 3-5x faster generation."
            )
        
        # Dependency recommendations
        deps = results["dependencies"]["versions"]
        if not deps.get("xformers"):
            recommendations.append(
                "Install xformers for memory-efficient attention."
            )
        
        return recommendations
    
    @staticmethod
    def print_diagnostic_summary(results: Dict[str, Any]):
        """Print a formatted summary of diagnostic results."""
        print("\n" + "=" * 60)
        print("FLUX DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        # System info
        print(f"\nSystem: {results['system']['platform']}")
        print(f"Python: {results['system']['python_version'].split()[0]}")
        print(f"RAM: {results['system']['total_ram_gb']:.1f}GB")
        
        # CUDA info
        cuda = results["cuda"]
        if cuda["available"]:
            print(f"\nCUDA: {cuda['version']}")
            for device in cuda["devices"]:
                print(f"GPU {device['index']}: {device['name']} ({device['total_memory_gb']:.1f}GB)")
        else:
            print("\nCUDA: Not available")
        
        # Issues
        issues = results["common_issues"]
        if issues:
            print("\nIssues Found:")
            for issue in issues:
                print(f"- [{issue['severity'].upper()}] {issue['issue']}")
                print(f"  {issue['details']}")
        else:
            print("\nNo major issues found!")
        
        # Recommendations
        if results["recommendations"]:
            print("\nRecommendations:")
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"{i}. {rec}")
        
        print("\n" + "=" * 60)


def diagnose_generation_failure(error: Exception, request: Any) -> Dict[str, Any]:
    """Diagnose a specific generation failure."""
    
    diagnosis = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "likely_cause": "Unknown",
        "suggested_fixes": []
    }
    
    error_str = str(error).lower()
    
    # Out of memory errors
    if "out of memory" in error_str or "oom" in error_str:
        diagnosis["likely_cause"] = "Insufficient GPU memory"
        diagnosis["suggested_fixes"] = [
            "Reduce image resolution",
            "Use a more quantized model (Q4 instead of Q8)",
            "Enable CPU offloading",
            "Free up GPU memory by closing other applications"
        ]
    
    # Device mismatch errors
    elif "expected all tensors to be on the same device" in error_str:
        diagnosis["likely_cause"] = "Device placement mismatch"
        diagnosis["suggested_fixes"] = [
            "Ensure all model components are on the same device",
            "Disable device_map='auto' for GGUF models",
            "Use manual device placement"
        ]
    
    # Model loading errors
    elif "model" in error_str and ("load" in error_str or "download" in error_str):
        diagnosis["likely_cause"] = "Model loading/download failure"
        diagnosis["suggested_fixes"] = [
            "Check internet connection",
            "Verify Hugging Face token if using gated models",
            "Clear model cache and retry",
            "Use a different model variant"
        ]
    
    # Timeout errors
    elif "timeout" in error_str or "stuck" in error_str:
        diagnosis["likely_cause"] = "Generation timeout"
        diagnosis["suggested_fixes"] = [
            "Reduce number of inference steps",
            "Use a faster model variant",
            "Check if GPU is being used (not CPU)",
            "Disable compile mode for first run"
        ]
    
    return diagnosis


def test_flux_setup():
    """Quick test to verify FLUX setup."""
    print("Testing FLUX setup...")
    
    try:
        # Test basic imports
        from diffusers import FluxPipeline
        print("✓ Diffusers imported successfully")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA not available")
        
        # Test memory
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU memory: {vram:.1f}GB")
        
        # Test quantization support
        try:
            from diffusers import GGUFQuantizationConfig
            print("✓ GGUF quantization support available")
        except ImportError:
            print("✗ GGUF quantization not available - update diffusers")
        
        print("\nSetup test complete!")
        
    except Exception as e:
        print(f"\n✗ Setup test failed: {e}")


if __name__ == "__main__":
    # Run diagnostics when module is executed directly
    FluxDiagnostics.run_full_diagnostic()