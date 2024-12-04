import cv2
import os
from typing import List, Tuple, Dict
import numpy as np

FILES: List[str] = os.listdir(os.path.join(os.getcwd(), "data"))

Matlike = np.ndarray

def filter_color(frame: Matlike) -> Matlike:
    """
    Filters the frame to isolate colors within the specified HSV range.

    Args:
        frame (Matlike): Input image in BGR format.

    Returns:
        Matlike: Image with only the filtered color in the specified HSV range.
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
    Combines two frames side by side into a single frame, allowing grayscale and color images.

    Args:
        frame1 (Matlike): The first frame (e.g., original frame).
        frame2 (Matlike): The second frame (e.g., processed frame).

    Returns:
        Matlike: A single frame with the two frames combined horizontally.
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
    Applies dilation to a thresholded binary image.

    Args:
        image (np.ndarray): Input binary or thresholded image.
        kernel_size (int): Size of the structuring element (default is 3).
        iterations (int): Number of times to apply dilation (default is 1).

    Returns:
        np.ndarray: Dilated image.
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
    Preprocess a frame by converting to grayscale and aplying a blut filter. Then, the frame is canny edge detected and dilated.

    Args:
        frame (Matlike): Input frame to be preprocessed.

    Returns:
        Matlike: Preprocessed frame.
    """
    filtered_frame = filter_color(frame)
    
    filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.blur(filtered_frame, (5, 5))
    
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    
    dilated = dilate_image(edges, iterations=1)
    
    return dilated

def draw_bbox(frame: Matlike, bbox: Tuple) -> Matlike:
    """
    Draws a bounding box on a frame.

    Args:
        frame (Matlike): Input frame.
        bbox (Tuple): Bounding box coordinates (x, y, width, height).
    
    Returns:
        Matlike: Frame with bounding box drawn.
    """
    x, y, w, h = bbox

    return cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (255, 0, 0), 2)


def get_components(frame: Matlike) -> Tuple[Matlike, Matlike, Matlike , Matlike]:
    """
    Get connected components in a binary image.

    Args:
        frame (Matlike): Input binary image.

    Returns:
        Tuple[Matlike, Matlike, Matlike , Matlike]:
            - num_labnels (int): Number of connected components.
            - labels (Matlike): Labels of connected components.
            - stats (Matlike): Statistics of connected components.
            - centroids (Matlike): Centroids of connected components.
    """
    components = cv2.connectedComponentsWithStats(frame, connectivity=8)
    return components

def filter_components(frame: Matlike, components: Tuple[Matlike, Matlike, Matlike , Matlike]) -> Tuple[Matlike, Tuple[int, int, int, int]]:
    """
    Detects connected components in a binary image, returns the bounding box coordinates of filtered components by area.

    Args:
        frame (Matlike): Input binary or grayscale image.
        components (Tuple[Matlike, Matlike, Matlike , Matlike]): Tuple of connected components.

    Returns:
        Tuple[Matlike, Tuple[int, int, int, int]]: 
            - Image on BGR colours.
            - Coordinates of bounding box of the largest connected component in (x, y, w, h) format.
    """
    num_labels, _, stats, _ = components
    output_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)

    dices: List[Tuple[int, int, int, int]] = []
    frame_area: int = frame.shape[0] * frame.shape[1]
    
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if abs(h / w - 1) <= 0.3 and area > 0.001 * frame_area and area < 0.003 * frame_area: 
            # cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dices.append((x, y, w, h))
    
    return output_frame, dices

def determine_dice(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, None]:
    """
    Determines the dice value based on the number of circles (dots) detected using Hough Circle.
    
    Args:
        frame (np.ndarray): The input image (BGR or grayscale).
        bbox (Tuple[int, int, int, int]): The bounding box (x, y, width, height) of the dice.

    Returns:
        int: The dice value (number of detected circles).
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
    """
    Draws dices on a frame.

    Args:
        frame (Matlike): Input frame.
        circles (List[Tuple[int, int, int]]): List of circles (x_center, y_center, radius).
        number (int, optional): Number of dices to draw. Defaults to None.
        border (int, optional): Border thickness. Defaults to 10.

    Returns:
        Matlike: Frame with dices drawn.
    """
    frame = frame.copy()
    
    if circles is None: return frame
    
    for circle in circles:
        x_center, y_center, radius = circle

        frame = cv2.circle(frame, (x_center, y_center), radius, (0, 255, 0), border)

    return frame

def write_image(frame: Matlike, point: Tuple[int, int], text: str, size=1) -> Matlike:
    """
    Writes text on a frame.

    Args:
        frame (Matlike): Input frame.
        point (Tuple[int, int]): Coordinates of the text position.
        text (str): Text to write.
        size (int, optional): Text size. Defaults to 1.

    Returns:
        Matlike: Frame with text written.
    """
    return cv2.putText(frame,
                        text,
                        (point[0], point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        size,
                        cv2.LINE_AA)


def get_green_region(components: Tuple[Matlike, Matlike, Matlike, Matlike]) -> Tuple[int, int, int, int] | None:
    """
    Returns the bounding box (x, y, width, height) of the green region with the largest area.

    Args:
    components : Tuple[Matlike, Matlike, Matlike, Matlike]
        The connected components of the frame.

    Returns:
    green_region : Tuple[int, int, int, int] or None
    
    The bounding box (x, y, width, height) of the largest green region, or None if no valid region is found.
    """
    _, _, stats, _ = components

    largest_area: int = 0
    largest_bbox = None

    for stat in stats:
        x, y, w, h, area = stat

        if area > 1000 and area > largest_area:  
            largest_area = area
            largest_bbox = (x, y, w, h)

    return largest_bbox
    

if __name__ == "__main__":
    results: List[Dict] = []
    
    for n_file, file in enumerate(FILES):
        video_path = os.path.join(os.getcwd(), "data", file)
        cap = cv2.VideoCapture(video_path)
        res = {i: 0 for i in range(5)}
        
        
        if not cap.isOpened():
            print(f"Could not open video file: {file}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = resize(frame)
            
            filtered_frame = preprocess(frame)
            
            components = get_components(filtered_frame)

            green_region = get_green_region(components)

            cv2.imshow("Frame", draw_bbox(frame, green_region))

            image_with_components, bboxes = filter_components(filtered_frame, components)
            
            if len(bboxes) == 5:            
                for n, dice in enumerate(bboxes):
                    number, circles = determine_dice(frame, dice)
                    sum_ = sum([x for x in res.values()])
                    
                    frame = draw_bbox(frame, dice)
                    frame = draw_dices(frame, circles, number, 10)
                    frame = write_image(frame, (dice[0] - 5, dice[1] - 5), f"N: {res[n]}")
                    frame = write_image(frame, (25, 25), f"Resultado: {sum_}", 3)
                    
                    res[n] = number if number > res[n] else res[n]
                    
            #cv2.imshow("Comparison", resize(combine_frames_side_by_side(frame, image_with_components)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        results.append(res)
        
        cap.release()
        cv2.destroyAllWindows()


    """
    Muestro los resultados
    """
    for k, res in enumerate(results):
        sum = 0
        
        print(f"Resultado {k+1}:")
        
        for i, num in res.items():
            print(f'\t- Dado {i}: {num}')
            sum += num
            
        print(f'\tResultado Final: {sum}')