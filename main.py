import cv2
import numpy
print("Versiune OpenCV" + cv2.__version__)

# Citire imagine
image_path = "example.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Debug imagine
########################################
# cv2.imshow("unprocessed image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Marimi imagine
(m, n) = image.shape

# Debug check
print("Nume imagine: " + image_path)
print("Marime imagine: " + str(m) + "x" + str(n))

from tv_isotropic import tv_isotropic, tv_iso_mat

print(tv_iso_mat(image))
