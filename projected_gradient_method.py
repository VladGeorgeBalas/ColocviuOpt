import gradient
import numpy as np

def ProjectedGradient(sursa, a, b):
    max_iter = 1000
    iters = 0
    ok = True
    image_matrix = np.copy(sursa)
    while (iters < max_iter) and ok:
        iters = iters + 1
        print("Iteratia: " + str(iters))

        alpha = 0.1

        # aplicam prima oara gradient lipschitz
        # aici alpha nu?? >:(
        new_image = image_matrix - alpha * gradient.gradient1_L(sursa, image_matrix)

        #aplicam subgradientul :[
        # momentan metoda banala cu un coeficient descrescator
        # NEAPARAT eficientizam numarul de iteratii cu un primal-dual splitting later
        # SI PE FIECARE A SI B IN PARTE CA E GRAV ALTFEl
        new_image = new_image - gradient.subgradient(new_image, a * alpha, b * alpha)

        #projection
        # proiectare gresita, se ia [a, b] si se proiecteaza pe [0, 255]
        # nu taiem din imagine ca ne trezim cu puncte negre sau albe, bai
        new_image = np.clip(new_image, 0, 255)

        #norm = np.linalg.norm(image_matrix - new_image, 'fro')
        norm = 1
        ok = ( norm > 1e-4 )
        image_matrix = new_image

    print ("iter = ", iters)
    print ("norm = ", norm)
    return image_matrix