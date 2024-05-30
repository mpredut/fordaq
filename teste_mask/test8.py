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

# Aplicarea unui prag pentru a obține o imagine binară
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Operații morfologice pentru a curăța imaginea binară
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Găsirea contururilor obiectului nou
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crearea unei măști pentru obiectul nou
mask = np.zeros_like(image1)

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

