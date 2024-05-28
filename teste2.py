import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para carregar e preprocessar a imagem
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# Função para calcular a projeção vertical
def calculate_vertical_projection(image):
    projection = np.sum(image, axis=1)
    return projection

# Função para aplicar blur (desfoque) ao histograma
def blur_histogram(histogram, kernel_size=9):
    kernel = np.ones(kernel_size) / kernel_size
    blurred_histogram = np.convolve(histogram, kernel, mode='same')
    return blurred_histogram

# Função para calcular a projeção horizontal
def calculate_horizontal_projection(image):
    projection = np.sum(image, axis=0)
    return projection

# Função principal para processar a imagem e detectar a placa
def process_image(image_path):
    gray_image = load_and_preprocess_image(image_path)

    # Aplicar filtro de Prewitt vertical
    prewitt_vertical = np.array([[1, 0, -1],
                                 [1, 0, -1],
                                 [1, 0, -1]])
    filtered_image = cv2.filter2D(gray_image, -1, prewitt_vertical)

    # Aplicar filtro de mediana
    filtered_image = cv2.medianBlur(filtered_image, 3)

    # Calcular histograma da intensidade das linhas (projeção vertical)
    vertical_projection = calculate_vertical_projection(filtered_image)
    normalized_histogram = blur_histogram(vertical_projection)

    # Limiarização (thresholding) para identificar os picos
    mean_intensity = np.mean(normalized_histogram)
    thresholded_histogram = np.where(normalized_histogram > mean_intensity, normalized_histogram, 0)

    # Descartar 20% das extremidades do histograma
    discard_percentage = 20
    discard_pixels = int(len(thresholded_histogram) * (discard_percentage / 100))
    thresholded_histogram[:discard_pixels] = 0
    thresholded_histogram[-discard_pixels:] = 0

    # Identificar pico de maior valor
    peak_index = np.argmax(thresholded_histogram)

    # Determinar intervalo candidato da região da placa
    plate_region_start = max(0, peak_index - 20)
    plate_region_end = min(len(gray_image), peak_index + 20)

    # Recortar a região da placa candidata
    plate_candidate = gray_image[plate_region_start:plate_region_end, :]

    # Aplicar filtro gaussiano
    blurred_plate = cv2.GaussianBlur(plate_candidate, (3, 3), 0)

    # Aplicar filtro de Prewitt horizontal
    prewitt_horizontal = np.array([[1, 1, 1],
                                   [0, 0, 0],
                                   [-1, -1, -1]])
    edges = cv2.filter2D(blurred_plate, -1, prewitt_horizontal)

    # Aplicar filtro de mediana
    edges = cv2.medianBlur(edges, 7)

    # Calcular a projeção horizontal
    horizontal_projection = calculate_horizontal_projection(edges)
    
    # Normalizar a projeção horizontal com filtro de média 15x15
    normalized_horizontal_projection = blur_histogram(horizontal_projection, 15)

    # Calcular a derivada da projeção horizontal
    h = 3
    derivative = np.diff(normalized_horizontal_projection, n=h)
    
    # Aplicar limiarização para detectar transições
    mean_derivative = np.mean(derivative)
    thresholded_derivative = np.where(derivative > mean_derivative, derivative, 0)

    # Descartar 10% das laterais da projeção
    discard_pixels_side = int(len(thresholded_derivative) * 0.1)
    thresholded_derivative[:discard_pixels_side] = 0
    thresholded_derivative[-discard_pixels_side:] = 0

    # Identificar picos de maior valor (14 picos de transição)
    peak_indices = np.argsort(thresholded_derivative)[-14:]
    peak_indices.sort()
    
    # Determinar intervalos candidatos dos caracteres da placa
    plate_char_start = peak_indices[0]
    plate_char_end = peak_indices[-1]
    
    # Adicionar margem de erro
    char_width = (plate_char_end - plate_char_start) // 7
    plate_char_start = max(0, plate_char_start - char_width)
    plate_char_end = min(len(plate_candidate[0]), plate_char_end + char_width)

    # Recortar a região da placa final
    final_plate = plate_candidate[:, plate_char_start:plate_char_end]

    # Mostrar imagem resultante com a região da placa candidata destacada
    plt.imshow(final_plate, cmap='gray')
    plt.title('Placa Candidata Final')
    plt.show()

# Substitua 'path_to_image' pelo caminho da sua imagem
process_image(r'Fotos\placa.jpeg')