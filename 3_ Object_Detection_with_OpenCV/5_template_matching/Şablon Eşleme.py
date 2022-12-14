# Template Matching

# Şablon eşleştirme, bir şablon görüntünün konumunu daha büyük bir
# görüntüde aramak ve bulmak için bir yöntemdir.

# Şablon görüntüsünü giriş görüntüsünün üzerine kaydırır ve şablon görüntüsünün
# altındaki giriş görüntüsünün şablonu ve yamayı karşılaştırır.

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpg", 0)
print(img.shape)

template = cv2.imread("cat_face.jpg", 0)
print("template.shape:", template.shape)

h, w = template.shape

# Gerekli metodları tanımlıyoruz (İki resim arasındaki korelasyonu inceler.)
methods = ["cv2.TM_CCOEFF", "cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR",
           "cv2.TM_CCORR_NORMED", "cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED"]

for meth in methods:

    method = eval(meth)  # "cv2.TM_CCOEFF" -> cv2.TM_CCOEFF

    # cv2.matchTemplate(image, templ(template), method)
    res = cv2.matchTemplate(img, template, method)
    # Yöntem çıktılarının boyutu aynı olmak zorunda!
    print(meth, res.shape)

    # Değer ve koordinatları alma işlemi
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.figure()

    plt.subplot(121)
    plt.imshow(res, cmap="gray")
    plt.title("Eşleşen Sonuç")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(img, cmap="gray")
    plt.title("Tespit Edilen Sonuç")
    plt.axis("off")

    plt.suptitle(meth.lstrip("cv2."))
    plt.show()
