import cv2
import numpy as np

def colaj(original, noise, org_grad_pro, org_grad_bar, org_grad_cvx, no_grad_pro, no_grad_bar, no_grad_cvx):
    colaj_sus = cv2.hconcat([original.astype("uint8"), org_grad_pro.astype("uint8"), org_grad_bar.astype("uint8"), org_grad_cvx.astype("uint8")])
    colaj_jos= cv2.hconcat([noise.astype("uint8"), no_grad_pro.astype("uint8"), no_grad_bar.astype("uint8"), no_grad_cvx.astype("uint8")])

    return cv2.vconcat([colaj_sus, colaj_jos])

def add_gaussian_noise(image, mean=0, var=100):
    """Add Gaussian noise to a grayscale image."""
    sigma = np.sqrt(var)
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss

    # Clip and convert to uint8
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy