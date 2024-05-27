import cv2
from matplotlib import pyplot as plt
import numpy as np

# Carregar a imagem
image = cv2.imread('carro.jpeg', 0)  # Carregar a imagem em escala de cinza

# Aplicar filtro de Prewitt vertical
prewitt_vertical = np.array([[1, 0, -1],
                              [1, 0, -1],
                              [1, 0, -1]])
filtered_image = cv2.filter2D(image, -1, prewitt_vertical)


# Aplicar filtro de mediana
filtered_image = cv2.medianBlur(filtered_image, 3)

# Calcular histograma da intensidade das linhas
histogram = np.sum(filtered_image, axis=1)


# Aplicar filtro de média 9x9 para normalização dos dados
normalized_histogram = cv2.blur(histogram, (9, 9))
plt.plot(normalized_histogram)
plt.show()

# Limiarização (thresholding) para identificar os picos
thresholded_histogram = np.where(normalized_histogram > 60, normalized_histogram, 0)

# Descartar 20% das extremidades do histograma
discard_percentage = 20
discard_pixels = int(len(thresholded_histogram) * (discard_percentage / 100))
thresholded_histogram[:discard_pixels] = 0
thresholded_histogram[-discard_pixels:] = 0

# Identificar pico de maior valor
peak_index = np.argmax(thresholded_histogram)

# Determinar intervalo candidato da região da placa
plate_region_start = max(0, peak_index - 20)
plate_region_end = min(len(image), peak_index + 20)

# Recortar a região da placa candidata
plate_candidate = image[plate_region_start:plate_region_end, :]

# Mostrar imagem resultante com a região da placa candidata destacada
cv2.imshow('Plate Candidate', plate_candidate)
cv2.waitKey(0)
cv2.destroyAllWindows()

