import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def try_gpu() -> None:
    logger.info("\n=== GPU Test for PyTorch ===")

    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")

    mps_available = torch.backends.mps.is_available()
    logger.info(f"MPS Available: {mps_available}")

    if cuda_available:
        device = torch.device("cuda")
        logger.info("Using CUDA device")

        device_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {device_count}")

        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            logger.info(
                f"Device {i}: {device_name} (Compute Capability: {device_capability[0]}.{device_capability[1]})"
            )

        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device}")
    elif mps_available:
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available. Using CPU")

    logger.info(f"\nRunning a simple tensor operation on {device}...")
    try:
        x = torch.rand(5, 3, device=device)
        y = torch.rand(5, 3, device=device)

        z = x + y

        logger.info("Tensor operation successful!")
        logger.info(f"Input tensor 1 shape: {x.shape}, device: {x.device}")
        logger.info(f"Input tensor 2 shape: {y.shape}, device: {y.device}")
        logger.info(f"Output tensor shape: {z.shape}, device: {z.device}")

        z_cpu = z.cpu()
        logger.info(f"Sample output values: {z_cpu[0]}")

        logger.info(f"\nTensor test on {device} completed successfully!")
    except Exception as e:
        logger.info(f"Error during tensor operation: {e}")

    if not (cuda_available or mps_available):
        logger.info("If you expected a GPU to be available, please check:")
        logger.info("For CUDA:")
        logger.info("1. You have an NVIDIA GPU")
        logger.info("2. NVIDIA drivers are installed")
        logger.info("3. CUDA toolkit is installed")
        logger.info("4. PyTorch was installed with CUDA support")
        logger.info("For MPS (Apple Silicon):")
        logger.info("1. You have an Apple M1/M2/M3/M4 chip")
        logger.info("2. You're using macOS 12.3+")
        logger.info("3. PyTorch 1.12+ is installed")

if __name__ == "__main__":
    try_gpu()