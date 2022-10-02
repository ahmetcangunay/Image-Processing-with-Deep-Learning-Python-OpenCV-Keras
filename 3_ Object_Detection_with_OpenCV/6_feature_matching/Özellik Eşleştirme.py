# Feature Matching

"""
-Görüntü işlemede nokta özelliği eşleşmesi, karmaşık bir sahnede belirtilen
bir hedefi tespit etmek için etkili bir yöntemdir.

-Bu yöntem, birden çok nesne yerine tek nesneleri algılar.

-Örneğin, bu yöntemi kullanarak, kişi dağınık bir görüntü üzerinde belirli bir
kişiyi tanıyabilir, ancak başka herhangi bir kişiyi tanıyamaz.

-Brute-Force eşleştiricisi, bir görüntüdeki bir özelliğin tanımlayıcısını başka
bir görüntünün diğer tüm özellikleriyle eşleştirir ve mesafeye göre eşleşmeyi
döndürür.

-Tüm özelliklerle eşleşmeyi kontrol ettiği için yavaştır.

-Ölçek değişmez özellik dönüşümü, anahtar noktaları ilk olarak bir dizi
referans görüntüden çıkarılır ve saklanır.

-Yeni görüntüdeki herbir özelliği bu saklanan veri ile ayrı ayrı
karşılaştırarak ve öznitelik vektörlerinin Öklid mesafesine dayalı olarak aday
eşleştirme özelliklerini bularak yeni bir görüntüde bir nesne tanınır.
"""
import cv2
import matplotlib.pyplot as plt

chos = cv2.imread("chocolates.jpg", 0)

plt.figure()
plt.imshow(chos, cmap="gray")
plt.axis("off")

cho = cv2.imread("nestle.jpg", 0)

plt.figure()
plt.imshow(cho, cmap="gray")
plt.axis("off")

# Orb tanımlayıcısı
# Köşe - kenar gibi nesneye ait özellikler belirlenir.
orb = cv2.ORB_create()

# Anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho, None)  # mask = None
kp2, des2 = orb.detectAndCompute(chos, None)

# Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Noktaları eşleştirme işlemi
matches = bf.match(des1, des2)

# Çok fazla eşleşme olacağından ötürü mesafeye göre sıralayıp belirli sayıda
# alacağız.
matches = sorted(matches, key=lambda x: x.distance)

# Eşleşen resimleri görselleştirelim.
plt.figure()

# cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg)
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags=2)

plt.imshow(img_match)
plt.title("Orb")
plt.axis("off")

# %% Sift kullanımı (Orb'dan daha iyi bir algoritmadır. Scale ve rotasyon
# farklılıklarında daha iyi çalışır.)

# Sift kullanabilmek için Console'da şu kodu çalıştırmak gerekir.
# pip install opencv-contrib-python --user

# Sift tanımlama işlemi
sift = cv2.xfeatures2d.SIFT_create()

# Brute-Force matcher
bf = cv2.BFMatcher()

# Anahtar nokta tespiti (Sift ile)
kp1, des1 = sift.detectAndCompute(cho, None)  # mask = None
kp2, des2 = sift.detectAndCompute(chos, None)  # mask = None

# Eşleşmeler
matches = bf.knnMatch(queryDescriptors=des1, trainDescriptors=des2, k=2)

# Matchesdeki ilk kısım güzel eşleşme olarak kabul edilir.
guzel_eslesme = []

for match1, match2 in matches:

    if match1.distance < 0.75*match2.distance:
        guzel_eslesme.append([match1])


plt.figure()

# cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches1to2, outImg)
sift_matches = cv2.drawMatchesKnn(
    cho, kp1, chos, kp2, guzel_eslesme, None, flags=2)

plt.imshow(sift_matches)
plt.title("Sift")
plt.axis("off")
