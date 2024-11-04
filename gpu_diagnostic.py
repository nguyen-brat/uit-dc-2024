import subprocess
import os
import torch
import sys
import time

def run_command(command):
    """Run shell command and return output."""
    try:
        result = subprocess.run(command, shell=True, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def check_gpu_health():
    """Comprehensive GPU health check."""
    results = {
        "gpu_info": {},
        "cuda_info": {},
        "memory_info": {},
        "errors": []
    }
    
    # Check NVIDIA driver and GPU status
    print("Checking GPU status...")
    nvidia_smi = run_command("nvidia-smi")
    if "NVIDIA-SMI" not in nvidia_smi:
        results["errors"].append("NVIDIA driver might not be properly installed")
    else:
        results["gpu_info"]["nvidia_smi"] = nvidia_smi

    # Check CUDA availability
    print("\nChecking CUDA...")
    results["cuda_info"]["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        results["cuda_info"]["cuda_version"] = torch.version.cuda
        results["cuda_info"]["gpu_count"] = torch.cuda.device_count()
        
        # Test CUDA memory allocation
        print("Testing CUDA memory allocation...")
        try:
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                # Try to allocate and free memory
                try:
                    x = torch.cuda.FloatTensor(1024, 1024)
                    del x
                    torch.cuda.empty_cache()
                    results["memory_info"][f"gpu_{i}_allocation"] = "Success"
                except RuntimeError as e:
                    results["memory_info"][f"gpu_{i}_allocation"] = f"Failed: {str(e)}"
        except Exception as e:
            results["errors"].append(f"Memory test error: {str(e)}")

    # Check GPU memory usage
    print("\nChecking memory usage...")
    try:
        for i in range(torch.cuda.device_count()):
            mem_info = torch.cuda.get_device_properties(i).total_memory
            mem_allocated = torch.cuda.memory_allocated(i)
            mem_cached = torch.cuda.memory_reserved(i)
            results["memory_info"][f"gpu_{i}_total"] = mem_info
            results["memory_info"][f"gpu_{i}_allocated"] = mem_allocated
            results["memory_info"][f"gpu_{i}_cached"] = mem_cached
    except Exception as e:
        results["errors"].append(f"Memory info error: {str(e)}")

    # Check NCCL configuration
    print("\nChecking NCCL configuration...")
    nccl_vars = {
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "Not set"),
        "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "Not set"),
        "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE", "Not set"),
        "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "Not set")
    }
    results["cuda_info"]["nccl_config"] = nccl_vars

    return results

def print_results(results):
    """Print diagnostic results in a readable format."""
    print("\n=== GPU Diagnostic Results ===")
    
    if results["gpu_info"].get("nvidia_smi"):
        print("\nGPU Status:")
        print(results["gpu_info"]["nvidia_smi"])
    
    print("\nCUDA Information:")
    print(f"CUDA Available: {results['cuda_info']['cuda_available']}")
    if results['cuda_info']['cuda_available']:
        print(f"CUDA Version: {results['cuda_info']['cuda_version']}")
        print(f"Number of GPUs: {results['cuda_info']['gpu_count']}")
    
    print("\nNCCL Configuration:")
    for key, value in results["cuda_info"]["nccl_config"].items():
        print(f"{key}: {value}")
    
    print("\nMemory Information:")
    for key, value in results["memory_info"].items():
        print(f"{key}: {value}")
    
    if results["errors"]:
        print("\nErrors Detected:")
        for error in results["errors"]:
            print(f"- {error}")

if __name__ == "__main__":
    print("Starting GPU diagnostics...")
    results = check_gpu_health()
    print_results(results)