import cv2
import numpy as np

# Citirea imaginilor
image1 = cv2.imread(f'cadre/{3}.jpg')
image2 = cv2.imread(f"cadre/{1}.jpg")

# Verificarea încărcării corecte a imaginilor
if image1 is None or image2 is None:
    print("Eroare la încărcarea imaginilor. Verificați căile fișierelor.")
    exit()

# Convertirea imaginilor la tonuri de gri
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Calcularea diferenței absolute între imagini
diff = cv2.absdiff(gray1, gray2)

# Aplicarea unui filtru median pentru reducerea zgomotului
diff_median = cv2.medianBlur(diff, 5)

# Aplicarea unui prag pentru a obține o imagine binară
_, thresh = cv2.threshold(diff_median, 30, 255, cv2.THRESH_BINARY)

# Operații morfologice pentru a curăța imaginea binară
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Aplicarea metodei de votare pentru clarificarea pixelilor
def voting_mask(mask, kernel_size=3):
    padded_mask = np.pad(mask, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant', constant_values=0)
    result_mask = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            neighborhood = padded_mask[i:i+kernel_size, j:j+kernel_size]
            if np.sum(neighborhood) > (kernel_size * kernel_size // 2) * 255:
                result_mask[i, j] = 255
    return result_mask

cleaned_mask = voting_mask(closing)

# Crearea unei măști pentru obiectul nou
mask = np.zeros_like(image1)

# Găsirea contururilor obiectului nou
contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extragem obiectul nou folosind contururile găsite
for contour in contours:
    if cv2.contourArea(contour) > 500:  # Filtrare contururi mici
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Aplicarea măștii pe imaginea color originală pentru a extrage obiectul nou
output = cv2.bitwise_and(image1, mask)

# Salvarea imaginii de ieșire
cv2.imwrite('obiect_nou_color.png', output)

# Afișarea imaginilor originale și a rezultatului
cv2.imshow('Diferenta', diff)
cv2.imshow('Obiect nou', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
