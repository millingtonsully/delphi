"""
GPU Verification Script for PyTorch

This script verifies that PyTorch can detect and use your NVIDIA GPU.
Run this after installing/updating your NVIDIA driver to confirm everything works.

Usage:
    python verify_gpu.py
"""

import sys
import torch
import time


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_cuda_availability():
    """Check if CUDA is available in PyTorch."""
    print_section("CUDA Availability Check")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ CUDA is not available!")
        print("\nPossible reasons:")
        print("  1. NVIDIA driver not installed or too old")
        print("  2. PyTorch was installed without CUDA support")
        print("  3. GPU not detected by system")
        print("\nNext steps:")
        print("  - Check Device Manager for GPU detection")
        print("  - Verify NVIDIA driver is installed (version 525.60.13+ for CUDA 12.1)")
        print("  - Run 'nvidia-smi' in terminal to test driver")
        return False
    
    print("✅ CUDA is available!")
    return True


def check_cuda_version():
    """Check CUDA version compatibility."""
    print_section("CUDA Version Information")
    
    # PyTorch's CUDA version
    pytorch_cuda_version = torch.version.cuda
    print(f"PyTorch CUDA Version: {pytorch_cuda_version}")
    
    # CUDA runtime version
    if torch.cuda.is_available():
        cuda_runtime_version = torch.version.cudnn if hasattr(torch.version, 'cudnn') else "N/A"
        print(f"cuDNN Version: {cuda_runtime_version}")
        
        # Expected CUDA version for PyTorch 2.5.1+cu121
        expected_cuda = "12.1"
        if pytorch_cuda_version and pytorch_cuda_version.startswith("12.1"):
            print(f"✅ CUDA version matches expected {expected_cuda}")
        else:
            print(f"⚠️  CUDA version may not match expected {expected_cuda}")
            print(f"   (PyTorch was built with CUDA {pytorch_cuda_version})")
    
    return True


def check_gpu_devices():
    """Check available GPU devices."""
    print_section("GPU Device Information")
    
    if not torch.cuda.is_available():
        print("No CUDA devices available.")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    
    if device_count == 0:
        print("❌ No GPU devices found!")
        return False
    
    print("\nGPU Details:")
    for i in range(device_count):
        print(f"\n  Device {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        
        # Get device properties
        props = torch.cuda.get_device_properties(i)
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"    Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"    Multiprocessors: {props.multi_processor_count}")
    
    print("\n✅ GPU devices detected successfully!")
    return True


def test_tensor_operations():
    """Test basic tensor operations on GPU."""
    print_section("GPU Tensor Operations Test")
    
    if not torch.cuda.is_available():
        print("❌ Cannot test tensor operations - CUDA not available")
        return False
    
    try:
        device = torch.device("cuda:0")
        print(f"Using device: {device}")
        
        # Test 1: Create tensor on GPU
        print("\nTest 1: Creating tensor on GPU...")
        x = torch.randn(1000, 1000, device=device)
        print(f"✅ Tensor created on GPU: {x.device}")
        print(f"   Shape: {x.shape}, Dtype: {x.dtype}")
        
        # Test 2: Perform computation
        print("\nTest 2: Performing matrix multiplication...")
        start_time = time.time()
        y = torch.matmul(x, x)
        torch.cuda.synchronize()  # Wait for GPU to finish
        elapsed = time.time() - start_time
        print(f"✅ Matrix multiplication completed in {elapsed:.4f} seconds")
        print(f"   Result shape: {y.shape}")
        
        # Test 3: Memory operations
        print("\nTest 3: Testing memory operations...")
        z = torch.zeros(5000, 5000, device=device)
        print(f"✅ Large tensor allocated on GPU: {z.shape}")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        print("✅ Memory cleaned up")
        
        print("\n✅ All tensor operations passed!")
        return True
        
    except RuntimeError as e:
        print(f"❌ Error during tensor operations: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_memory_allocation():
    """Test GPU memory allocation."""
    print_section("GPU Memory Allocation Test")
    
    if not torch.cuda.is_available():
        print("❌ Cannot test memory allocation - CUDA not available")
        return False
    
    try:
        device = torch.device("cuda:0")
        
        # Get memory info
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
        memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        
        print(f"GPU Memory Status:")
        print(f"  Total: {memory_total:.2f} GB")
        print(f"  Reserved: {memory_reserved:.2f} GB")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Free: {memory_total - memory_reserved:.2f} GB")
        
        # Try allocating a reasonable amount of memory
        print("\nTesting memory allocation...")
        test_tensor = torch.randn(1000, 1000, device=device)
        allocated_after = torch.cuda.memory_allocated(device) / (1024**3)
        print(f"✅ Successfully allocated tensor")
        print(f"   Memory after allocation: {allocated_after:.2f} GB")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        
        print("✅ Memory allocation test passed!")
        return True
        
    except RuntimeError as e:
        print(f"❌ Error during memory allocation: {e}")
        return False


def test_performance_comparison():
    """Compare GPU vs CPU performance (if GPU available)."""
    print_section("Performance Comparison (GPU vs CPU)")
    
    if not torch.cuda.is_available():
        print("Skipping performance comparison - CUDA not available")
        return True
    
    try:
        size = 2000
        iterations = 10
        
        # CPU test
        print(f"Running CPU test ({iterations} iterations, {size}x{size} matrices)...")
        x_cpu = torch.randn(size, size)
        start = time.time()
        for _ in range(iterations):
            y_cpu = torch.matmul(x_cpu, x_cpu)
        cpu_time = time.time() - start
        
        # GPU test
        print(f"Running GPU test ({iterations} iterations, {size}x{size} matrices)...")
        device = torch.device("cuda:0")
        x_gpu = torch.randn(size, size, device=device)
        torch.cuda.synchronize()  # Warm up
        start = time.time()
        for _ in range(iterations):
            y_gpu = torch.matmul(x_gpu, x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        print(f"\nResults:")
        print(f"  CPU time: {cpu_time:.4f} seconds")
        print(f"  GPU time: {gpu_time:.4f} seconds")
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 1:
            print(f"✅ GPU is {speedup:.2f}x faster than CPU!")
        else:
            print(f"⚠️  GPU performance may be limited (check driver/thermal throttling)")
        
        # Clean up
        del x_cpu, y_cpu, x_gpu, y_gpu
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"⚠️  Performance comparison failed: {e}")
        return True  # Don't fail overall if this test fails


def main():
    """Run all GPU verification tests."""
    print("\n" + "=" * 70)
    print("  PyTorch GPU Verification Script")
    print("=" * 70)
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    results = []
    
    # Run all tests
    results.append(("CUDA Availability", check_cuda_availability()))
    
    if results[0][1]:  # Only continue if CUDA is available
        results.append(("CUDA Version", check_cuda_version()))
        results.append(("GPU Devices", check_gpu_devices()))
        results.append(("Tensor Operations", test_tensor_operations()))
        results.append(("Memory Allocation", test_memory_allocation()))
        results.append(("Performance Comparison", test_performance_comparison()))
    else:
        print("\n⚠️  Skipping remaining tests - CUDA not available")
    
    # Summary
    print_section("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your GPU is ready for PyTorch training.")
        print("\nYou can now run your training script:")
        print("  python -m delphi.training.train --config configs/delphi_config.yaml")
        return 0
    elif results[0][1]:  # CUDA available but some tests failed
        print("\n⚠️  Some tests failed. GPU may have issues but basic functionality works.")
        return 1
    else:
        print("\n❌ GPU not detected. Please install/update NVIDIA driver.")
        print("\nDriver installation guide:")
        print("  1. Download driver from: https://www.nvidia.com/Download/index.aspx")
        print("  2. Select: GeForce GTX 1050, Windows 10/11, 64-bit")
        print("  3. Install with 'Custom (Advanced)' -> 'Perform a clean installation'")
        print("  4. Restart computer after installation")
        print("  5. Run this script again to verify")
        return 1


def print_gpu_monitoring_tips():
    """Print tips for monitoring GPU usage during training."""
    print_section("How to Monitor GPU Usage During Training")
    
    print("\nThere are several ways to verify your GPU is being used during training:\n")
    
    print("1. CHECK TRAINING LOGS:")
    print("   - The training script now automatically prints GPU memory usage")
    print("   - Look for 'GPU Status' and 'GPU Memory' messages at the start of training")
    print("   - Memory usage should increase when training starts\n")
    
    print("2. USE NVIDIA-SMI (Recommended for real-time monitoring):")
    print("   Open a separate terminal/command prompt and run:")
    print("   - Watch continuously: nvidia-smi -l 1")
    print("   - Or check once: nvidia-smi")
    print("   Look for:")
    print("     • GPU-Util: Should be high (50-100%) during training")
    print("     • Memory-Usage: Should show memory being used")
    print("     • Processes: Should show your Python process using GPU\n")
    
    print("3. CHECK TENSOR DEVICES IN CODE:")
    print("   The trainer now verifies tensors are on GPU automatically")
    print("   You can also manually check:")
    print("     tensor.device  # Should show 'cuda:0' or similar")
    print("     next(model.parameters()).device  # Check model device\n")
    
    print("4. PERFORMANCE INDICATORS:")
    print("   - GPU training should be noticeably faster than CPU")
    print("   - If using CPU, you'll see '⚠️  Using CPU' message")
    print("   - GPU memory usage should increase with batch size\n")
    
    print("5. WINDOWS TASK MANAGER:")
    print("   - Open Task Manager (Ctrl+Shift+Esc)")
    print("   - Go to 'Performance' tab")
    print("   - Select GPU")
    print("   - Monitor 'GPU Utilization' and 'Dedicated GPU Memory'\n")
    
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--monitoring-tips":
        print_gpu_monitoring_tips()
    else:
        sys.exit(main())

