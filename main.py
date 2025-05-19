import cv2
import numpy
import projected_gradient_method
import matplotlib.pyplot as plt
from barrier_gradient_method import BarrierGradient
from projected_gradient_method import ProjectedGradient
print("Versiune OpenCV" + cv2.__version__)

# Citire imagine
image_path = "Poze/example7.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Debug imagine
#######################################
cv2.imshow("Read image", image)
# Marimi imagine
(m, n) = image.shape

# Debug check
print("Nume imagine: " + image_path)
print("Marime imagine: " + str(m) + "x" + str(n))

from tv_isotropic import tv_isotropic, tv_iso_mat
from cvx_grad import cvx_solve

# image = tv_iso_mat(image)
# image = gradient(255 * numpy.ones((m, n)), image, 0.1, 0.1)
(image_pro, err_pro) = ProjectedGradient(image, 2, 0.1)
plt.plot(numpy.log(err_pro), label="Projected Gradient")
plt.show()
cv2.imshow("Gradient Method", image_pro.astype("uint8"))
#(image_bar, err_bar) = BarrierGradient(image, 2, 0.1)
#plt.plot(numpy.log(err_bar), label="Barrier Gradient")
#plt.show()
# image_cvx = cvx_solve(image, 2, 1)

# Debug imagine
#######################################
#cv2.imshow("Barier Method", image_bar.astype("uint8"))
# cv2.imshow("CVX Method", image_cvx.astype("uint8"))
# cv2.imwrite(image_path + "tv_iso.jpg", (1/10 * image).astype("uint8"))
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Debug imagine
# #######################################
# print(image)