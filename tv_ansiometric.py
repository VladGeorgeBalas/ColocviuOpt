import numpy
from scipy.ndimage import shift


def tv_aniso_mat(image_matrix):
    (m, n) = image_matrix.shape
    eps = 0

    image_matrix = image_matrix.astype("double")

    mat11 = numpy.sign(
        image_matrix - shift(image_matrix, (-1, 0), cval=0)
    )

    mat12 = numpy.sign(
        image_matrix - shift(image_matrix, (0, -1), cval=0)
    )

    mat2 = (-1) * numpy.sign(
        (
                image_matrix - shift(image_matrix, (1, 0), cval=0)
        )
    )

    mat3 = (-1) * numpy.sign(
        (
                image_matrix - shift(image_matrix, (0, 1), cval=0)
        )
    )

    result = numpy.zeros((m, n), dtype="double")
    for i in range(0, m - 1):
        for j in range(0, n - 1):
            result[i, j] = (mat11[i, j] if mat11[i, j] != 0 else eps) + (mat12[i, j] if mat12[i, j] != 0 else eps) + (
                mat2[i, j] if mat2[i, j] != 0 else eps) + (mat3[i, j] if mat3[i, j] != 0 else eps)

    return result
