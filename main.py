import cv2
import numpy
import projected_gradient_method
import matplotlib.pyplot as plt
from barrier_gradient_method import BarrierGradient
from projected_gradient_method import ProjectedGradient
from cvx_grad import cvx_solve
from colaj import colaj
from colaj import add_gaussian_noise
print("Versiune OpenCV" + cv2.__version__)

# Citire imagine
image_path = "Poze/example7.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print("Path: " + image_path)
print("Image shape: " + str(image.shape))

noisy_image = add_gaussian_noise(image)

(image_pro, err_pro) = ProjectedGradient(noisy_image, 2, 0.1)
(image_bar, err_bar) = BarrierGradient(noisy_image, 2, 0.1)
image_cvx = cvx_solve(noisy_image, 2, 0.1)

result = colaj(image, noisy_image, image_pro, image_bar, image_cvx)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

from tv_isotropic import tv_isotropic, tv_iso_mat


# image = tv_iso_mat(image)
# image = gradient(255 * numpy.ones((m, n)), image, 0.1, 0.1)
# plt.plot(numpy.log(err_pro), label="Projected Gradient")
# plt.show()
# cv2.imshow("Gradient Method", image_pro.astype("uint8"))
#plt.plot(numpy.log(err_bar), label="Barrier Gradient")
#plt.show()

from colaj import colaj, add_gaussian_noise



# Debug imagine
#######################################
# cv2.imshow("Barier Method", image_bar.astype("uint8"))
# cv2.imshow("CVX Method", image_cvx.astype("uint8"))
# cv2.imwrite(image_path + "tv_iso.jpg", (1/10 * image).astype("uint8"))


# # Debug imagine
# #######################################
# print(image)