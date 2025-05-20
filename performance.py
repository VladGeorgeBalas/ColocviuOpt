import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_mse(imageA, imageB):
    return np.mean((imageA.astype(np.float64) - imageB.astype(np.float64)) ** 2)

def compute_psnr_grayscale(imageA, imageB):
    return psnr(imageA, imageB, data_range=255)

def compute_ssim_grayscale(imageA, imageB):
     ssim_value, _ = ssim(imageA, imageB, full=True, data_range=255)
     return ssim_value

def print_performance(imageA, imageB):
    mse = compute_mse(imageA, imageB)
    psnr = compute_psnr_grayscale(imageA, imageB)
    ssim = compute_ssim_grayscale(imageA, imageB)
    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")