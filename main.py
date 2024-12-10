import cv2
import os
from typing import List, Tuple, Dict
import numpy as np

FILES: List[str] = os.listdir(os.path.join(os.getcwd(), "data"))

Matlike = np.ndarray

def crop_green_region(frame: Matlike) -> Matlike:
    """
    Crops the green region from the frame using the mask.
    Args:
        frame (Matlike): The frame to be cropped.
        mask (Matlike): The mask to be used for cropping.
    Returns:
        Matlike: The cropped image.
    """

    lower_black = np.array([0, 0, 0])      # Lower bound of black
    upper_black = np.array([180, 255, 50]) # Upper bound of black
    mask = cv2.inRange(frame, lower_black, upper_black)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = frame[y:y+h, x:x+w]
        return cropped_image
    return None



def filter_color(frame: Matlike) -> Matlike:
    """
    Filtra el frame para aislar los colores dentro del rango HSV especificado.

    Args:
        frame (Matlike): Imagen de entrada en formato BGR.

    Returns:
        Matlike: Imagen con solo el color filtrado dentro del rango HSV especificado.
    """
    hMin, sMin, vMin = 0, 0, 0
    hMax, sMax, vMax = 30, 255, 255
    lower_bound = np.array([hMin, sMin, vMin])
    upper_bound = np.array([hMax, sMax, vMax])

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return filtered_frame

def combine_frames_side_by_side(frame1: Matlike, frame2: Matlike) -> Matlike:
    """
    Combina dos frames lado a lado en un solo frame, permitiendo imágenes en escala de grises y color.

    Args:
        frame1 (Matlike): El primer frame (por ejemplo, el frame original).
        frame2 (Matlike): El segundo frame (por ejemplo, el frame procesado).

    Returns:
        Matlike: Un solo frame con los dos frames combinados horizontalmente.
    """
    if len(frame1.shape) == 2:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
    
    if len(frame2.shape) == 2:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

    height1, width1, _ = frame1.shape
    height2, width2, _ = frame2.shape

    max_height = max(height1, height2)
    scale1 = max_height / height1
    scale2 = max_height / height2

    resized_frame1 = cv2.resize(frame1, (int(width1 * scale1), max_height))
    resized_frame2 = cv2.resize(frame2, (int(width2 * scale2), max_height))

    combined_frame = np.hstack((resized_frame1, resized_frame2))

    return combined_frame

def resize(frame: Matlike) -> Matlike:
    return cv2.resize(frame.copy(),
                      (frame.shape[1]//2, frame.shape[0]//2))

def dilate_image(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Aplica dilatación a una imagen binaria umbralizada.

    Args:
        image (np.ndarray): Imagen binaria o umbralizada de entrada.
        kernel_size (int): Tamaño del elemento estructurante (por defecto es 3).
        iterations (int): Número de veces que se aplica la dilatación (por defecto es 1).

    Returns:
        np.ndarray: Imagen dilatada.
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    dilated_image = cv2.dilate(binary_image, kernel, iterations=iterations)

    return dilated_image

def preprocess(frame: Matlike) -> Matlike:
    """
    Preprocesa un frame convirtiéndolo a escala de grises y aplicando un filtro de desenfoque. Luego, se aplica la detección de bordes Canny y dilatación.

    Args:
        frame (Matlike): Frame de entrada a preprocesar.

    Returns:
        Matlike: Frame preprocesado.
    """
    filtered_frame = filter_color(frame)
    
    filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.blur(filtered_frame, (5, 5))
    
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    
    dilated = dilate_image(edges, iterations=1)
    
    return dilated, edges, blurred

def draw_bbox(frame: Matlike, bbox: Tuple) -> Matlike:
    """
    Dibuja un cuadro delimitador en un frame.

    Args:
        frame (Matlike): Frame de entrada.
        bbox (Tuple): Coordenadas del cuadro delimitador (x, y, ancho, alto).
    
    Returns:
        M
    """
    x, y, w, h = bbox

    return cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (255, 0, 0), 2)

def filter_components(frame: Matlike) -> Tuple[Matlike, Tuple[int, int, int, int]]:
    """
    Detecta componentes conectados en una imagen binaria,
    y devuelve las coordenadas del cuadro delimitador de los componentes filtrados por área.

    Args:
        frame (Matlike): Imagen binaria o en escala de grises de entrada.

    Returns:
        Tuple[Matlike, Tuple[int, int, int, int]]: 
            - Imagen en colores BGR.
            - Coordenadas del cuadro delimitador del componente conectado más grande en formato (x, y, w, h).
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(frame, connectivity=8)
    output_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)

    dices: List[Tuple[int, int, int, int]] = []
    frame_area: int = frame.shape[0] * frame.shape[1]
    
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if abs(h / w - 1) <= 0.3 and area > 0.001 * frame_area and area < 0.003 * frame_area: 
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dices.append((x, y, w, h))
    
    return output_frame, dices

def determine_dice(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, None]:
    """
    Determina el valor del dado basado en el número de círculos (puntos) detectados usando el algoritmo de Círculos de Hough.

    Args:
        frame (np.ndarray): Imagen de entrada (en formato BGR o escala de grises).
        bbox (Tuple[int, int, int, int]): El cuadro delimitador (x, y, ancho, alto) del dado.

    Returns:
        int: El valor del dado (número de círculos detectados).
    """
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    gray_roi = cv2.resize(gray_roi, (gray_roi.shape[1]*4, gray_roi.shape[0]*4))
    
    blurred_roi = cv2.blur(gray_roi, (12, 12))
    
    circles = cv2.HoughCircles(blurred_roi,
                               cv2.HOUGH_GRADIENT,
                               dp=0.03,
                               minDist=2,
                               param1=60,
                               param2=31,
                               minRadius=0,
                               maxRadius=0)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        new_circles = []
        
        for (x_center, y_center, radius) in circles:
            new_circles.append((x_center + frame.shape[0], y_center + frame.shape[1], radius))
        
        return len(new_circles), new_circles
    
    return 0, None

def draw_dices(frame: Matlike, circles: List[Tuple[int, int, int]], number: int = None, border: int = 10) -> Matlike:
    frame = frame.copy()
    
    if circles is None: return frame
    
    for circle in circles:
        x_center, y_center, radius = circle

        frame = cv2.circle(frame, (x_center, y_center), radius, (0, 255, 0), border)

    return frame

def write_image(frame: Matlike, point: Tuple[int, int], text: str, size=1) -> Matlike:
    return cv2.putText(frame,
                        text,
                        (point[0], point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        size,
                        cv2.LINE_AA)

def export_video(output_path: str,
                 frame_size: tuple,
                 fps: int,
                 frame_generator) -> str:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame in frame_generator:
        frame_resized = cv2.resize(frame, frame_size)
        out.write(frame_resized)

    out.release()
    
    print(f"Video guardado en: {output_path}")

if __name__ == "__main__":
    """
    El buffer se crea para poder guardar posteriormente los videos.
    """
    results: List[Dict] = []
    buffer: Dict[str, Dict[str, List]] = {os.path.basename(file).split('.')[0] : {'filtered': [],
                                                                                  'components': [],
                                                                                  'final' : [],
                                                                                  'edges' : [],
                                                                                  'blurred' : []} for file in FILES}
    
    for n_file, file in enumerate(FILES):
        filename: str = os.path.basename(file).split('.')[0]
        video_path = os.path.join(os.getcwd(), "data", file)
        cap = cv2.VideoCapture(video_path)
        res = {i: {'number': 0, 'last_pos':-1} for i in range(5)}
        
        if not cap.isOpened():
            print(f"Could not open video file: {file}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = resize(frame)

            frame = crop_green_region(frame)
            #cv2.imshow("Cropped Region", cropped_region)
            
            filtered_frame, edges, blurred = preprocess(frame)
            
            image_with_components, bboxes = filter_components(filtered_frame)
            
            if len(bboxes) == 5:            
                for n, dice in enumerate(bboxes):
                    number, circles = determine_dice(frame, dice)
                    sum_ = sum([x['number'] for x in res.values()])
                    
                    if abs(res[n]['last_pos'] - dice[0] + [2])[0] <= 10:
                        frame = draw_dices(frame, circles, number, 10)
                        
                        frame = write_image(frame, (dice[0] - 5, dice[1] - 5), f"N: {res[n]['number']}")
                        
                        frame = write_image(frame, (25, 25), f"Resultado: {sum_}", 3)
                        
                        frame = draw_bbox(frame, dice)
                    
                    res[n] = {
                        'number' : number if number > res[n]['number'] else res[n]['number'],
                        'last_pos' : dice[0] + [2]
                        }
                    
            buffer[filename]['filtered'].append(cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR))
            buffer[filename]['edges'].append(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
            buffer[filename]['blurred'].append(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR))
            buffer[filename]['components'].append(image_with_components)
            buffer[filename]['final'].append(frame)
            
            cv2.imshow("Comparison", resize(combine_frames_side_by_side(frame, image_with_components)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        results.append(res)
        
        cap.release()
        cv2.destroyAllWindows()

    """
    Exporto todos los videos.
    """
    for file in buffer.keys():
        images = buffer[file]
        
        for image, buffer_ in images.items():
            path = os.path.join(os.getcwd(), 'output', f'{file}-{image}.avi')
            export_video(output_path=path,
                         frame_size=(360, 741),
                         fps=30,
                         frame_generator=iter(buffer_))

    """
    Muestro los resultados
    """
    for k, res in enumerate(results):
        sum = 0
        
        print(f"Resultado {k+1}:")
        
        for i, num in res.items():
            number = num['number']
            print(f'\t- Dado {i}: {number}')
            sum += number
            
        print(f'\tResultado Final: {sum}')
