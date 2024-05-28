import cv2
from matplotlib import pyplot as plt
import numpy as np

# Carregar a imagem
def projecao_vertical(image):  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro de Prewitt vertical
    prewitt_vertical = np.array([[1, 0, -1],
                                [1, 0, -1],
                                [1, 0, -1]])
    filtered_image = cv2.filter2D(image, -1, prewitt_vertical)


    # Aplicar filtro de mediana
    filtered_image = cv2.medianBlur(filtered_image, 3)

    # Calcular histograma da intensidade das linhas
    histogram = np.sum(filtered_image, axis=1)
    plt.plot(histogram)
    plt.show()

    # Aplicar filtro de média 9x9 para normalização dos dados
    #normalized_histogram = cv2.blur(histogram, (9, 9)) SUPOSTAMENTE FUNCIONA NO MAC
    kernel_size = 9
    kernel = np.ones(kernel_size) / kernel_size
    normalized_histogram = np.convolve(histogram, kernel, mode='same')


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
    return plate_candidate

    # Mostrar imagem resultante com a região da placa candidata destacada
    cv2.imshow('Plate Candidate', plate_candidate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_plate_image(image):
    # Aplicar filtro gaussiano 3x3 para suavizar a imagem
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Aplicar o filtro de Prewitt no eixo horizontal
    kernel_prewitt_x = np.array([[1, 0, -1], 
                                 [1, 0, -1], 
                                 [1, 0, -1]], dtype=np.float32)
    edges_prewitt_x = cv2.filter2D(blurred, -1, kernel_prewitt_x)

    # Aplicar filtro de mediana 7x7 para remover ruídos
    edges_prewitt_x_median = cv2.medianBlur(edges_prewitt_x, 7)

    # Projeção vertical da imagem (soma dos pixels em cada coluna)
    vertical_projection = np.sum(edges_prewitt_x_median, axis=0)

    # Normalizar a projeção vertical com um filtro de média 15x15
    kernel_average = np.ones(15) / 15
    smoothed_projection = np.convolve(vertical_projection, kernel_average, mode='same')

    # Calcular a derivada da projeção vertical
    h = 3
    derivative_projection = np.diff(smoothed_projection, n=h)

    # Encontrar picos (transições de preto para branco e branco para preto)
    peaks_black_to_white = np.where(derivative_projection > 0)[0]
    peaks_white_to_black = np.where(derivative_projection < 0)[0]

    # Combinar picos e ordenar
    peaks = np.sort(np.concatenate((peaks_black_to_white, peaks_white_to_black)))

    # Descartar 10% das laterais da projeção
    margin = int(0.1 * len(vertical_projection))
    peaks = peaks[(peaks > margin) & (peaks < len(vertical_projection) - margin)]

    # Identificar 14 maiores picos de intensidade
    if len(peaks) >= 14:
        significant_peaks = peaks[np.argsort(smoothed_projection[peaks])[-14:]]

        # Ordenar os picos encontrados
        significant_peaks = np.sort(significant_peaks)

        # Delimitar a região candidata da placa
        candidate_plate_start = significant_peaks[0]
        candidate_plate_end = significant_peaks[-1]

        # Adicionar margem de erro baseada na largura da região dividida por 7
        margin_error = (candidate_plate_end - candidate_plate_start) // 7
        candidate_plate_start -= margin_error
        candidate_plate_end += margin_error

        # Garantir que os índices estão dentro dos limites da imagem
        candidate_plate_start = max(0, candidate_plate_start)
        candidate_plate_end = min(len(vertical_projection), candidate_plate_end)

        # Extrair a região candidata da placa
        candidate_plate = image[:, candidate_plate_start:candidate_plate_end]

        # Exibir a imagem original e a região candidata
        return candidate_plate
    else:
        print("Não foram encontrados picos suficientes para identificar a placa.")



image = cv2.imread(r'Fotos\placa.webp')
image = projecao_vertical(image)
cv2.imshow('image', image)
cv2.waitKey(0)

# image = cv2.imread(r'Fotos\APENASAPLACA.jpeg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = process_plate_image(image)
# cv2.imshow('image', image)
# cv2.waitKey(0)




