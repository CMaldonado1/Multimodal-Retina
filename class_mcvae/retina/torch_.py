import torch

def main():
    if torch.cuda.is_available():
        print(f"CUDA is available with {torch.cuda.device_count()} GPU(s)!")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert bytes to GB
            print(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f} GB")
    else:
        print("CUDA is not available. Exiting.")

if __name__ == "__main__":
    main()

