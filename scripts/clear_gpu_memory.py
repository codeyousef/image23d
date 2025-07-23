#!/usr/bin/env python3
"""
Clear GPU memory and show before/after stats
"""

import torch
import gc
import subprocess
import time

def get_gpu_memory_info():
    """Get current GPU memory info"""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return {
            "total_gb": total / (1024**3),
            "free_gb": free / (1024**3),
            "used_gb": used / (1024**3),
            "percent_used": (used / total) * 100
        }
    return None

def clear_gpu_memory():
    """Clear GPU memory"""
    print("Clearing GPU memory...")
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Clear any remaining PyTorch allocations
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def kill_gpu_processes():
    """Kill processes using GPU (optional, requires confirmation)"""
    try:
        # Get GPU processes
        result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            print("\nProcesses using GPU:")
            lines = result.stdout.strip().split('\n')
            
            processes = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    pid = parts[0]
                    process_name = parts[1]
                    memory = parts[2]
                    print(f"  PID {pid}: {process_name} - {memory}")
                    processes.append((pid, process_name))
            
            if processes:
                response = input("\nKill these processes? (y/N): ").lower()
                if response == 'y':
                    for pid, name in processes:
                        try:
                            if 'python' in name.lower():
                                print(f"Killing {name} (PID: {pid})...")
                                subprocess.run(['kill', '-9', pid])
                        except Exception as e:
                            print(f"Failed to kill {pid}: {e}")
                    time.sleep(2)  # Wait for processes to die
                    
    except Exception as e:
        print(f"Error checking GPU processes: {e}")

def main():
    print("GPU Memory Cleaner")
    print("=" * 50)
    
    # Show before stats
    before = get_gpu_memory_info()
    if before:
        print(f"\nBefore clearing:")
        print(f"  Total: {before['total_gb']:.2f} GB")
        print(f"  Used: {before['used_gb']:.2f} GB ({before['percent_used']:.1f}%)")
        print(f"  Free: {before['free_gb']:.2f} GB")
    else:
        print("No GPU detected!")
        return
    
    # Clear memory
    clear_gpu_memory()
    
    # Option to kill processes
    if before['free_gb'] < 10:  # If less than 10GB free
        print(f"\nWarning: Only {before['free_gb']:.2f} GB free!")
        kill_gpu_processes()
        clear_gpu_memory()  # Clear again after killing
    
    # Show after stats
    after = get_gpu_memory_info()
    print(f"\nAfter clearing:")
    print(f"  Total: {after['total_gb']:.2f} GB")
    print(f"  Used: {after['used_gb']:.2f} GB ({after['percent_used']:.1f}%)")
    print(f"  Free: {after['free_gb']:.2f} GB")
    
    # Show improvement
    freed = after['free_gb'] - before['free_gb']
    if freed > 0:
        print(f"\nFreed: {freed:.2f} GB")
    else:
        print("\nNo additional memory freed (may need to close other applications)")
    
    print("\n" + "=" * 50)
    print("Done!")

if __name__ == "__main__":
    main()