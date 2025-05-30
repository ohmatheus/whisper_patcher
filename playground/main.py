import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cuda_gpu() -> None:
    logger.info("\n=== CUDA GPU Test for PyTorch ===")

    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")

    if cuda_available:
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

        logger.info("\nRunning a simple tensor operation on GPU...")
        try:
            x = torch.rand(5, 3).cuda()
            y = torch.rand(5, 3).cuda()

            z = x + y

            logger.info("Tensor operation successful!")
            logger.info(f"Input tensor 1 shape: {x.shape}, device: {x.device}")
            logger.info(f"Input tensor 2 shape: {y.shape}, device: {y.device}")
            logger.info(f"Output tensor shape: {z.shape}, device: {z.device}")

            z_cpu = z.cpu()
            logger.info(f"Sample output values: {z_cpu[0]}")

            logger.info("\nCUDA GPU test completed successfully!")
        except Exception as e:
            logger.info(f"Error during tensor operation: {e}")
    else:
        logger.info("CUDA is not available. PyTorch will run on CPU only.")
        logger.info("If you expected CUDA to be available, please check:")
        logger.info("1. You have an NVIDIA GPU")
        logger.info("2. NVIDIA drivers are installed")
        logger.info("3. CUDA toolkit is installed")
        logger.info("4. PyTorch was installed with CUDA support")


def main() -> None:
    logger.info("Whisper Sound Bank - Starting application...")

    test_cuda_gpu()

    # TODO: Play with Whisper transcription


if __name__ == "__main__":
    main()
