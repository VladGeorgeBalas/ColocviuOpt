from tv_ansiometric import tv_aniso_mat
from tv_isotropic import  tv_iso_mat

def gradient(sursa, image_matrix, a, b):
    return 2 * (sursa - image_matrix) + a * tv_iso_mat(image_matrix) + b * tv_aniso_mat(image_matrix)