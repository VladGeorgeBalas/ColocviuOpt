import cv2
import numpy
from skimage.metrics import peak_signal_noise_ratio

import projected_gradient_method
import matplotlib.pyplot as plt
from barrier_gradient_method import BarrierGradient
from performance import compute_psnr_grayscale, compute_ssim_grayscale
from projected_gradient_method import ProjectedGradient
from cvx_grad import cvx_solve
from colaj import colaj
from colaj import add_gaussian_noise
from performance import print_performance
print("Versiune OpenCV" + cv2.__version__)

# Citire imagine
image_path = "Poze/example13.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print("Path: " + image_path)
print("Image shape: " + str(image.shape))


noisy_image = add_gaussian_noise(image)

print("Imagine originala vs Imagine Noisy")
print_performance(image, noisy_image)

(org_pro, err_org_pro) = ProjectedGradient(image, 2, 0.1)
(org_bar, err_org_bar) = BarrierGradient(image, 2, 0.1)
org_cvx = cvx_solve(image, 2, 0.1)

print("Imagine originala vs Imagine originala_proiectie")
print_performance(image, org_pro)

print("Imagine originala vs Imagine originala_bariera")
print_performance(image, org_bar)

print("Imagine originala vs Imagine originala_cvx")
print_performance(image, org_cvx)

(no_pro, err_no_pro) = ProjectedGradient(noisy_image, 2, 0.1)
(no_bar, err_no_bar) = BarrierGradient(noisy_image, 2, 0.1)
noo_cvx = cvx_solve(noisy_image, 2, 0.1)

print("Imagine originala vs Imagine noisy_proiectie")
print_performance(image, no_pro)

print("Imagine originala vs Imagine noisy_bariera")
print_performance(image, no_bar)

print("Imagine originala vs Imagine noisy_cvx")
print_performance(image, noo_cvx)

print("Imagine originala proectie vs Imagine noisy proiectie")
print_performance(org_pro, no_pro)

plt.plot(numpy.log10(err_org_pro), label="Projected Gradient On Original Image")
plt.xlabel("Iteration")
plt.ylabel("Log Error")
plt.title("Projected Gradient Convergence")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(numpy.log10(err_org_bar), label="Barrier Gradient On Original Image")
plt.xlabel("Iteration")
plt.ylabel("Log Error")
plt.title("Barrier Gradient Convergence")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(numpy.log10(err_no_pro), label="Projected Gradient On Noisy Image")
plt.xlabel("Iteration")
plt.ylabel("Log Error")
plt.title("Projected Gradient Convergence")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(numpy.log10(err_no_bar), label="Barrier Gradient On Noisy Image")
plt.xlabel("Iteration")
plt.ylabel("Log Error")
plt.title("Barrier Gradient Convergence")
plt.legend()
plt.grid(True)
plt.show()

result = colaj(image, noisy_image, org_pro, org_bar, org_cvx, no_pro, no_bar, noo_cvx)
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