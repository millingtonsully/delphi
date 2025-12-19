"""
Real-time GPU monitoring script for training.

Run this in a separate terminal while training to monitor GPU usage in real-time.

Usage:
    python monitor_gpu.py
    python monitor_gpu.py --watch 2  # Update every 2 seconds (default: 1)
"""

import sys
import time
import subprocess
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Only showing nvidia-smi output.")


def get_nvidia_smi():
    """Get GPU status from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_pytorch_gpu_info():
    """Get GPU info from PyTorch."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    
    info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        info.append({
            'device': i,
            'name': torch.cuda.get_device_name(i),
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': total - reserved
        })
    return info


def format_nvidia_smi_output(output):
    """Format nvidia-smi output for display."""
    if not output:
        return None
    
    lines = output.strip().split('\n')
    formatted = []
    formatted.append("┌─────────────────────────────────────────────────────────────────┐")
    formatted.append("│ GPU Status (from nvidia-smi)                                    │")
    formatted.append("├─────────────────────────────────────────────────────────────────┤")
    
    for line in lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 6:
            idx, name, util, mem_used, mem_total, temp = parts
            mem_pct = (int(mem_used) / int(mem_total)) * 100 if mem_total != '0' else 0
            formatted.append(f"│ GPU {idx}: {name[:30]:<30} │")
            formatted.append(f"│   Utilization: {util:>3}%  Memory: {mem_used:>6}/{mem_total:>6} MB ({mem_pct:>5.1f}%) │")
            formatted.append(f"│   Temperature: {temp}°C                                          │")
            formatted.append("├─────────────────────────────────────────────────────────────────┤")
    
    formatted[-1] = "└─────────────────────────────────────────────────────────────────┘"
    return '\n'.join(formatted)


def format_pytorch_output(info_list):
    """Format PyTorch GPU info for display."""
    if not info_list:
        return None
    
    formatted = []
    formatted.append("┌─────────────────────────────────────────────────────────────────┐")
    formatted.append("│ GPU Memory (from PyTorch)                                       │")
    formatted.append("├─────────────────────────────────────────────────────────────────┤")
    
    for info in info_list:
        formatted.append(f"│ GPU {info['device']}: {info['name'][:30]:<30} │")
        formatted.append(f"│   Allocated: {info['allocated']:>6.2f} GB  Reserved: {info['reserved']:>6.2f} GB │")
        formatted.append(f"│   Free: {info['free']:>6.2f} GB  Total: {info['total']:>6.2f} GB          │")
        formatted.append("├─────────────────────────────────────────────────────────────────┤")
    
    formatted[-1] = "└─────────────────────────────────────────────────────────────────┘"
    return '\n'.join(formatted)


def main():
    """Main monitoring loop."""
    import argparse
    parser = argparse.ArgumentParser(description='Monitor GPU usage in real-time')
    parser.add_argument('--watch', type=float, default=1.0,
                       help='Update interval in seconds (default: 1.0)')
    args = parser.parse_args()
    
    print("=" * 65)
    print("  GPU Monitoring Tool")
    print("=" * 65)
    print(f"Updating every {args.watch} seconds...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Clear screen (works on Windows and Unix)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 65)
            print(f"  GPU Status - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 65)
            print()
            
            # nvidia-smi output
            nvidia_output = get_nvidia_smi()
            if nvidia_output:
                formatted = format_nvidia_smi_output(nvidia_output)
                if formatted:
                    print(formatted)
                    print()
            else:
                print("⚠️  nvidia-smi not available. Make sure NVIDIA drivers are installed.")
                print()
            
            # PyTorch output
            if TORCH_AVAILABLE:
                pytorch_info = get_pytorch_gpu_info()
                if pytorch_info:
                    formatted = format_pytorch_output(pytorch_info)
                    if formatted:
                        print(formatted)
                        print()
                else:
                    print("⚠️  CUDA not available in PyTorch")
                    print()
            else:
                print("⚠️  PyTorch not installed")
                print()
            
            print("=" * 65)
            print(f"Next update in {args.watch} seconds... (Press Ctrl+C to stop)")
            
            time.sleep(args.watch)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()










