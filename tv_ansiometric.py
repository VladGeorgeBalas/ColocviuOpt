import numpy as np
from scipy.ndimage import shift

def tv_aniso_mat(image_matrix):
    image_matrix = image_matrix.astype(np.float64)

    diff_up = image_matrix - shift(image_matrix, (-1, 0), mode='nearest')
    diff_left = image_matrix - shift(image_matrix, (0, -1), mode='nearest')
    diff_down = image_matrix - shift(image_matrix, (1, 0), mode='nearest')
    diff_right = image_matrix - shift(image_matrix, (0, 1), mode='nearest')

    # Signs of differences
    sign_up = np.sign(diff_up)
    sign_left = np.sign(diff_left)
    sign_down = np.sign(diff_down)
    sign_right = np.sign(diff_right)

    # Sum all contributions
    result = sign_up + sign_left + sign_down + sign_right

    return result
