import cv2
import numpy as np

def colaj(original, noise, grad_pro, grad_bar, cvx):
    colaj = cv2.hconcat([original.astype("uint8"), noise.astype("uint8"), grad_pro.astype("uint8"), grad_bar.astype("uint8"), cvx.astype("uint8")])

    return colaj

def add_gaussian_noise(image, mean=0, var=100):
    """Add Gaussian noise to a grayscale image."""
    sigma = np.sqrt(var)
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss

    # Clip and convert to uint8
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy