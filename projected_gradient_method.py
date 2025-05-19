import gradient
import numpy as np
import numpy

def ProjectedGradient(sursa, a, b):
    max_iter = 100
    iters = 0
    ok = True
    image_matrix = np.copy(sursa).astype("float32")
    err = []
    while (iters < max_iter) and ok:
        iters = iters + 1
        print("Iteratia: " + str(iters))

        alpha = 0.15
        beta = 1
        # aplicam prima oara gradient lipschitz
        new_image = image_matrix - alpha * gradient.gradient1_L(sursa, image_matrix)

        #aplicam subgradientul :[
        # momentan metoda banala cu un coeficient descrescator
        # NEAPARAT eficientizam numarul de iteratii cu un primal-dual splitting later
        # SI PE FIECARE A SI B IN PARTE CA E GRAV ALTFEl
        new_image = new_image - beta * gradient.subgradient(new_image, a, b)

        #projection
        # proiectare gresita, se ia [a, b] si se proiecteaza pe [0, 255]
        # nu taiem din imagine ca ne trezim cu puncte negre sau albe, bai

        # noua proiectie, trebuie eficientizata
        min = numpy.min(new_image)
        if min < 0:
            new_image = new_image + numpy.ones(new_image.shape) * (-min)
        max = numpy.max(new_image)
        if max > 255:
            new_image = new_image * ( 255 / max)

        # proiectie veche
        # new_image = np.clip(new_image, 0, 255)

        norm = np.linalg.norm(image_matrix - new_image, 'fro')
        err.append(norm)
        ok = ( norm > 1e-3 )
        image_matrix = new_image

    print ("iter = ", iters)
    print ("norm = ", norm)
    return (image_matrix, err)