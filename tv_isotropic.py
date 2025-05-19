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
    import cv2
    (m, n) = image_matrix.shape

    base = numpy.sqrt(
            numpy.power(image_matrix - shift(image_matrix, (-1,0), cval=0), 2) +
            numpy.power(image_matrix - shift(image_matrix, (0,-1), cval=0), 2)
    )

    mat1 = (
            image_matrix - shift(image_matrix, (-1,0), cval=0) +
            image_matrix - shift(image_matrix, (0,-1), cval=0)
    )

    mat2 = (
        image_matrix - shift(image_matrix, (1, 0), cval=0)
    )

    mat3 = (
            image_matrix - shift(image_matrix, (0, 1), cval=0)
    )

    # Create shifted views with padding to handle edges
    inv_base = numpy.zeros_like(base, dtype=float)
    nonzero = base != 0
    inv_base[nonzero] = 1.0 / base[nonzero]

    result = numpy.zeros((m, n), dtype=float)

    # Current cell
    result[:-1, :-1] += inv_base[:-1, :-1] * mat1[:-1, :-1]

    # Top neighbor
    result[1:, :-1] += inv_base[:-1, :-1] * mat2[1:, :-1]

    # Left neighbor
    result[:-1, 1:] += inv_base[:-1, :-1] * mat3[:-1, 1:]


    #result = numpy.zeros((m, n))
    #for i in range(0, m - 1):
     #   for j in range(0, n - 1):
      #      if base[i, j] != 0:
       #         result[i, j] += (1 / base[i, j]) * mat1[i, j]
        #    if i > 0 and base[i - 1, j] != 0:
         #       result[i, j] += (1 / base[i - 1, j]) * mat2[i, j]
          #  if j > 0 and base[i, j - 1] != 0:
           #     result[i, j] += (1 / base[i, j - 1]) * mat3[i, j]
    # cv2.imshow("base", base.astype('uint8'))
    # cv2.imshow("mat1", mat1.astype('uint8'))
    # cv2.imshow("mat2", mat2.astype('uint8'))
    # cv2.imshow("mat3", mat3.astype('uint8'))

    return result