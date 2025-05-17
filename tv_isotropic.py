# tv(x) = sum_{i,j}(sqrt((x(i,j) - x(i + 1,j))^2 + (x(i,j) - x(i,j + 1))^2)

# pentru derivare luam
# f1 = sqrt((x(i,j) - x(i + 1,j))^2 + (x(i,j) - x(i,j + 1))^2
# f2 = sqrt((x(i - 1,j) - x(i,j))^2 + (x(i - 1,j) - x(i - 1,j + 1))^2
# f3 = sqrt((x(i,j - 1) - x(i + 1,j - 1))^2 + (x(i,j - 1) - x(i,j))^2

# si derivata lui tv(x) in functie de x(i,j) va fi suma lui f1, f2 si f3 derivate, acolo unde ele exista

import numpy

def tv_isotropic(image_matrix):
    (m, n) = image_matrix.shape
    result = numpy.zeros((m, n))
    epsilon = 1e-8

    def f1d(x, i, j):
        if (x[i, j] - x[i + 1, j]) ** 2 + (x[i, j] - x[i, j + 1])**2 == 0:
            return 0
        else:
            result = (1 / numpy.sqrt((x[i, j] - x[i + 1, j]) ** 2 + (x[i, j] - x[i, j + 1]) ** 2)) * (
                2 * (x[i, j] - x[i + 1, j]) + 2 * (x[i, j] - x[i, j + 1]))
            return result


    def f2d(x, i, j):
        if (x[i - 1, j] - x[i, j]) ** 2 + (x[i - 1, j] - x[i - 1, j + 1]) ** 2 == 0:
            return 0
        else:
            result = (1 / numpy.sqrt((x[i - 1, j] - x[i, j]) ** 2 + (x[i - 1, j] - x[i - 1, j + 1]) ** 2)) * (
                x[i, j] - x[i - 1, j])
            return result


    def f3d(x, i, j):
        if (x[i, j - 1] - x[i + 1, j - 1]) ** 2 + (x[i, j - 1] - x[i, j]) ** 2 == 0:
            return 0
        else:
            result = (1 / numpy.sqrt((x[i, j - 1] - x[i + 1, j - 1]) ** 2 + (x[i, j - 1] - x[i, j]) ** 2)) * (
                x[i, j] - x[i, j - 1])
            return result

    for i in range(0, m - 1):
        for j in range(0, n - 1):
            result[i, j] += f1d(image_matrix, i, j) if (i < m - 1 and j < n - 1) else 0
            result[i, j] += f2d(image_matrix, i, j) if (i > 0 and j < n - 1) else 0
            result[i, j] += f3d(image_matrix, i, j) if (i < m - 1 and j > 0) else 0

    return result

def tv_iso_mat(image_matrix):
    from scipy.ndimage import shift
    (m, n) = image_matrix.shape
    result = numpy.power(image_matrix - shift(image_matrix, (-1,0), cval=0), 2) + numpy.power(image_matrix - shift(image_matrix, (0,-1), cval=0), 2)

    return result