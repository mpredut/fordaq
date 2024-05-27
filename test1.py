import cv2
import numpy as np
import os

def lipire_cadre_orizontal_cu_masca(folder_imagini, folder_masti, numar_cadre, imagine_output):
    cadre = []
    masti = []

    # Încărcăm cadrele și măștile
    for i in range(2, numar_cadre + 1):  # începem de la 2 pentru că 1 nu are scândură
        img_path = os.path.join(folder_imagini, f"{i}.jpg")
        mask_path = os.path.join(folder_masti, f"mask_{i}.png")

        if not os.path.exists(img_path):
            print(f"Error: Image {img_path} does not exist.")
            continue

        if not os.path.exists(mask_path):
            print(f"Error: Mask {mask_path} does not exist.")
            continue

        cadru = cv2.imread(img_path)
        masca = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if cadru is None:
            print(f"Error: Could not load image {img_path}.")
            continue

        if masca is None:
            print(f"Error: Could not load mask {mask_path}.")
            continue

        cadre.append(cadru)
        masti.append(masca)

    if len(cadre) == 0 or len(masti) == 0:
        print("Error: No valid images or masks were loaded.")
        return

    # Dimensiunile unui cadru
    inaltime, latime, _ = cadre[0].shape

    # Calculăm lățimea totală a imaginii finale
    latime_totala = 0
    for idx, masca in enumerate(masti):
        _, x_indices = np.where(masca > 0)
        if x_indices.size == 0:
            continue
        x_min, x_max = x_indices.min(), x_indices.max()
        latime_totala += (x_max - x_min + 1)

    # Crearea unei pânze suficient de mari pentru a stoca imaginea finală
    inaltime_tablou = inaltime
    tablou_final = np.zeros((inaltime_tablou, latime_totala, 3), dtype=cadre[0].dtype)

    current_x = 0

    # Lipim cadrele în ordine inversă
    for idx, (cadru, masca) in enumerate(zip(cadre[::-1], masti[::-1])):
        # Determinăm marginile scândurii folosind masca
        y_indices, x_indices = np.where(masca > 0)

        if y_indices.size == 0 or x_indices.size == 0:
            print(f"Warning: No mask found in image at index {numar_cadre - idx}.")
            continue

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Extragem partea relevantă a scândurii
        scindura = cadru[y_min:y_max + 1, x_min:x_max + 1]

        # Plasăm partea relevantă a scândurii pe pânza finală
        tablou_final[0:scindura.shape[0], current_x:current_x + scindura.shape[1]] = scindura
        current_x += scindura.shape[1]

    # Eliminăm orice spații negre la sfârșitul imaginii
    output_image = tablou_final[:, :current_x]  # Decupăm pânza pentru a elimina zona nefolosită

    if output_image.size == 0:
        print("Error: Output image is empty.")
        return

    # Salvarea imaginii finale
    success = cv2.imwrite(imagine_output, output_image)
    if not success:
        print(f"Error: Could not write the image to {imagine_output}.")
    else:
        print(f"Image successfully saved to {imagine_output}.")

# Exemplu de utilizare
lipire_cadre_orizontal_cu_masca("cadre", "mask", 19, "scandura_completa_compact.jpg")
