#!/usr/bin/env python3
"""
Diagnose GPU memory usage and find what's consuming VRAM
"""

import torch
import subprocess
import psutil
from pathlib import Path

def get_gpu_info():
    """Get detailed GPU information"""
    print("=== GPU Information ===")
    
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            
            # Memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1024**3
            
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Free: {free:.2f} GB")
    else:
        print("CUDA not available!")

def get_gpu_processes():
    """Get processes using GPU via nvidia-smi"""
    print("\n=== GPU Process List ===")
    
    try:
        # Run nvidia-smi to get process list
        result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            print("Processes using GPU:")
            lines = result.stdout.strip().split('\n')
            total_used = 0
            
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    pid = parts[0]
                    process_name = parts[1]
                    memory = parts[2]
                    print(f"  PID {pid}: {process_name} - {memory}")
                    
                    # Try to parse memory usage
                    if 'MiB' in memory:
                        mb = float(memory.replace(' MiB', ''))
                        total_used += mb / 1024  # Convert to GB
            
            print(f"\nTotal GPU Memory Used by Processes: {total_used:.2f} GB")
        else:
            print("No processes found using GPU")
            
    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

def check_wsl_memory():
    """Check WSL memory limits if running in WSL"""
    print("\n=== WSL Memory Check ===")
    
    # Check if running in WSL
    if Path("/proc/version").exists():
        with open("/proc/version", "r") as f:
            version = f.read()
            if "microsoft" in version.lower():
                print("Running in WSL")
                
                # Check .wslconfig
                wslconfig_path = Path.home() / ".wslconfig"
                if wslconfig_path.exists():
                    print(f"\n.wslconfig found at: {wslconfig_path}")
                    with open(wslconfig_path, "r") as f:
                        print("Contents:")
                        print(f.read())
                else:
                    print("\nNo .wslconfig found - using default WSL memory settings")
                    print("WSL2 by default uses up to 50% of total RAM")
                    
                # Check current memory
                mem = psutil.virtual_memory()
                print(f"\nWSL Memory:")
                print(f"  Total: {mem.total / 1024**3:.2f} GB")
                print(f"  Available: {mem.available / 1024**3:.2f} GB")
                print(f"  Used: {mem.percent:.1f}%")
            else:
                print("Not running in WSL")
    else:
        print("Not running in WSL")

def check_zombie_processes():
    """Check for zombie PyTorch processes"""
    print("\n=== Checking for Zombie Processes ===")
    
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                mem_gb = proc.info['memory_info'].rss / 1024**3
                python_processes.append((proc.info['pid'], proc.info['name'], mem_gb))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if python_processes:
        print("Python processes:")
        for pid, name, mem in python_processes:
            print(f"  PID {pid}: {name} - {mem:.2f} GB RAM")
    else:
        print("No Python processes found")

def suggest_fixes():
    """Suggest fixes based on findings"""
    print("\n=== Suggested Fixes ===")
    
    print("1. Clear GPU memory:")
    print("   python -c \"import torch; torch.cuda.empty_cache()\"")
    
    print("\n2. Kill all Python processes:")
    print("   pkill -f python")
    
    print("\n3. Restart NVIDIA driver (Windows):")
    print("   Run as admin: net stop nvlddmkm && net start nvlddmkm")
    
    print("\n4. Check for memory leaks in other apps:")
    print("   - Chrome/Edge with hardware acceleration")
    print("   - Discord, OBS, or other GPU-accelerated apps")
    print("   - Other AI/ML applications")
    
    print("\n5. For WSL users:")
    print("   - Restart WSL: wsl --shutdown")
    print("   - Increase GPU memory limit in .wslconfig")

def main():
    print("GPU Memory Diagnostic Tool")
    print("=" * 50)
    
    get_gpu_info()
    get_gpu_processes()
    check_wsl_memory()
    check_zombie_processes()
    suggest_fixes()
    
    print("\n" + "=" * 50)
    print("Diagnostic complete!")

if __name__ == "__main__":
    main()