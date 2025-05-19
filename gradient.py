from tv_ansiometric import tv_aniso_mat
from tv_isotropic import  tv_iso_mat
import numpy as np

def subgradient(image_matrix, a, b):
    return  a * tv_iso_mat(image_matrix) + b * tv_aniso_mat(image_matrix)

def gradient1_L(sursa, image_matrix):
    return (image_matrix - sursa)
# am despartit in gradient lipshitz  ( L = 2 )si subgradient

def barrier_gradient(image_matrix):
    epsilon = 1e-3
    X = np.clip(image_matrix, epsilon, 255 - epsilon)
    grad = -1 / X + 1 / (255 - X)
    grad[image_matrix <= 0] = -1e4
    grad[image_matrix >= 255] = 1e4
    return grad
