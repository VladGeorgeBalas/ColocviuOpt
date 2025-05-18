import cv2
import numpy
import projected_gradient_method
from barrier_gradient_method import BarrierGradient
from projected_gradient_method import ProjectedGradient
print("Versiune OpenCV" + cv2.__version__)

# Citire imagine
image_path = "example2_noise.jpg"
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

# image = tv_iso_mat(image)
# image = gradient(255 * numpy.ones((m, n)), image, 0.1, 0.1)
# image = ProjectedGradient(image, 2, 0.1)
image = BarrierGradient(image, 2, 0.1)

# Debug imagine
#######################################
cv2.imshow("Gradient Method", image.astype("uint8"))
# cv2.imwrite(image_path + "tv_iso.jpg", (1/10 * image).astype("uint8"))
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Debug imagine
# #######################################
# print(image)