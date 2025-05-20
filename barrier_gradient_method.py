import gradient
import numpy as np

def BarrierGradient(sursa, a, b):
    max_iter = 200
    iters = 0
    ok = True
    image_matrix = np.copy(sursa)
    bar_coef = 8e-1
    err = []
    while (iters < max_iter) and ok:
        iters = iters + 1
        # aplicam prima oara gradient cu pas lipschitz * alpha
        alpha = 1
        new_image = image_matrix -alpha * gradient.gradient1_L(sursa, image_matrix)

        #aplicam subgradientul TV
        beta = 5
        new_image = new_image - gradient.subgradient(new_image, a * beta, b * beta)

        #aplicam barieria
        bar_coef = bar_coef  * 0.7
        new_image = new_image - bar_coef * gradient.barrier_gradient(new_image)

        norm = np.linalg.norm(image_matrix - new_image, 'fro')
        err.append(norm)
        ok = ( norm > 1e-6 )
        image_matrix = new_image

    #print ("iter = ", iters)
    print ("barrier norm = ", norm)
    #print (image_matrix[image_matrix > 255])
    image_matrix = np.clip(image_matrix, 0, 255)
    return (image_matrix, err)