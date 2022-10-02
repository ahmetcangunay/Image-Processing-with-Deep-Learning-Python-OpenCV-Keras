# Bulanıklaştırma İşleminde Low Pass Filter Kullanılır.
# Yüksek Frekanslı Gürültüleri (Parazit ve Kenarlar) kaldırılır.

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Sistemin vereceği uyarıları kaldırma
import warnings

warnings.filterwarnings("ignore")

# Blurring (Detay Azaltma ve Gürültü Engelleme)
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title("Original")
plt.show()

# %% Average Blurring Method

# Kutucukların içindeki tüm değerlerin ortalaması alınarak merkeze yazılır.

# cv2.blur(src, ksize = Kutucuk Boyutu)
# Çıktılar "dst", Girdiler "src" olarak dökümantasyonda adlandırılır.
dst2 = cv2.blur(img, (3, 3))

plt.figure()
plt.imshow(dst2)
plt.axis("off")
plt.title("Average Blurring Method")
plt.show()

# %% Gaussian Blurring Method

# Gauss Noise'u elemek için kullanılan yöntemdir.
# Kutucuklar "kernel" olarak adlandırılır.
# X ve Y yönlerinde sigma değerleri yazarak kutucukların 2D gauss olmasını sağlıyoruz.

# cv2.GaussianBlur(src, ksize = kernel size (Kutucuk boyutu), sigmaX = x yönündeki sigma)
# Eğer y yönündekini belirtmezsek x'e eşit olur.

gb = cv2.GaussianBlur(img, (3, 3), 7)

plt.figure()
plt.imshow(gb)
plt.axis("off")
plt.title("Gaussian Blurring Method")
plt.show()

# %% Median Blurring Method

# Seçilen kutucukların medyan değerindeki genlik değeri alınır. Merkezin değeri olarak yazılır.
# cv2.medianBlur(src, ksize = Kernel Size)
mb = cv2.medianBlur(img, 3)

plt.figure()
plt.imshow(mb)
plt.axis("off")
plt.title("Median Blurring Method")
plt.show()


# %% Noise Oluşturma

def gaussian_noise(image):
    # Channel (ch) bir resmin gray mi, RGB mi yoksa BGR mı olduğunu belirtir.

    row, col, ch = image.shape  # (512,512,3) gibi
    mean = 0  # Ortalama
    var = 0.05  # Varyans
    sigma = var ** 0.5  # Standart Sapma

    gauss = np.random.normal(mean, sigma, (row, col, ch))  # Gaussian (Normal) dağılım
    gauss = gauss.reshape(row, col, ch)  # Boyutun uyumlu olduğundan emin olduk.
    noisy_img = image + gauss

    return noisy_img


# Normalize Ederek İçe Aktarma

img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255  # Değerler 0 ile 1 arasında olmalıdır.

plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title("Original")
plt.show()

gNoisyImg = gaussian_noise(img)

plt.figure()
plt.imshow(gNoisyImg)
plt.axis("off")
plt.title("Gaussian Noisy Image")
plt.show()

# %% Gaussian Noise Azaltma (Gaussian Blur)

gb2 = cv2.GaussianBlur(gNoisyImg, (3, 3), 7)

plt.figure()
plt.imshow(gb2)
plt.axis("off")
plt.title("Gaussian Blurring Method Last")
plt.show()


# %% Salt Pepper Noise

def salt_pepper_noise(image):
    row, col, ch = image.shape
    s_vs_p = 0.3

    amount = 0.004

    noisy = np.copy(image)

    # Salt
    num_salt = np.ceil(amount * image.size * s_vs_p)  # 1800, 1900 gibi değerler beklenir.
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords] = 1

    # Peppers
    num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))  # 1800, 1900 gibi değerler beklenir.
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords] = 0

    return noisy


spImg = salt_pepper_noise(img)

plt.figure()
plt.imshow(spImg)
plt.axis("off")
plt.title("Salt & Pepper")
plt.show()

# Bu tip noiselerden kurtulmak için "Median Blur" kullanacağız.
mb2 = cv2.medianBlur(spImg.astype(np.float32), 3)

plt.figure()
plt.imshow(mb2)
plt.axis("off")
plt.title("Salt & Pepper Last Method")
plt.show()
