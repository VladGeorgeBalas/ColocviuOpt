import gradient
import numpy as np

def BarrierGradient(sursa, a, b):
    max_iter = 100
    iters = 0
    ok = True
    image_matrix = np.copy(sursa)
    bar_coef = 1e-1
    while (iters < max_iter) and ok:
        iters = iters + 1
        # aplicam prima oara gradient lipschitz
        new_image = image_matrix - gradient.gradient1_L(sursa, image_matrix)

        #aplicam subgradientul TV
        # momentan metoda banala cu un coeficient descrescator
        # NEAPARAT eficientizam numarul de iteratii cu un primal-dual splitting later
        alpha = 5
        new_image = new_image - gradient.subgradient(new_image, a * alpha, b * alpha)

        #aplicam barieria
        bar_coef = bar_coef  * 0.98
        new_image = new_image - bar_coef * gradient.barrier_gradient(new_image)

        norm = np.linalg.norm(image_matrix - new_image, 'fro')
        ok = ( norm > 1e-6 )
        image_matrix = new_image

    print ("iter = ", iters)
    print ("norm = ", norm)
    print (image_matrix[image_matrix > 255])
    return image_matrix