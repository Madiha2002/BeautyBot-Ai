import cv2
import numpy as np
from .landmarks import detect_landmarks, normalize_landmarks, plot_landmarks
from mediapipe.python.solutions.face_detection import FaceDetection

upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 408, 415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76]
lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
face_conn = [10, 338, 297, 332, 284, 251, 389, 264, 447, 376, 433, 288, 367, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 138, 213, 147, 234, 127, 162, 21, 54, 103, 67, 109]
left_cheeks = [234, 50]
right_cheeks = [454, 280]

# 425
left_cheekbone = [93, 132, 207]  # Adjust landmarks as needed
right_cheekbone = [ 361,323, 427]  # Adjust landmarks as needed

left_forehead = [ 139, 54, 103, 67 , 109 ]
right_forehead = [338,297,332, 251 ,301]


# work foundation
face_contour = [10, 338, 297, 332, 284, 251, 389, 264, 447, 376, 433, 288, 367, 397, 365, 379, 378, 400, 377, 152,
                148, 176, 149, 150, 136, 172, 138, 213, 147, 234, 127, 162, 21, 54, 103, 67, 109]

left_eye = [33, 133, 160, 159, 158, 144, 145, 153, 173, 157, 158, 163, 153]

right_eye = [263, 362, 387, 386, 385, 373, 374, 380, 384, 385, 373, 388, 390]

lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

def apply_makeup_on_bytes(image_bytes: bytes,data, is_stream: bool, features: list, show_landmarks: bool = False,) -> bytes:
    """
    Applies makeup effects to an image provided as bytes and returns the result as bytes.
    
    :param image_bytes: Image data in bytes.
    :param is_stream: Boolean indicating if the input is from a video stream.
    :param features: List of features to apply makeup to (e.g., ['lips', 'blush', 'foundation']).
    :param show_landmarks: Whether to display landmarks for the processed feature.
    :return: Processed image as bytes.
    """
    # Decode the image bytes to a NumPy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    src = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if src is None:
        raise ValueError("Could not decode the image bytes.")

    # Process the image using the existing logic
    ret_landmarks = detect_landmarks(src, is_stream)

    # Check if landmarks were detected; if not, return the original image as bytes
    if ret_landmarks is None:
        print("No face detected")
        _, result_bytes = cv2.imencode('.jpg', src)
        return result_bytes.tobytes()

    height, width, _ = src.shape
    output = src.copy()
    all_masks = np.zeros_like(src)

    # Loop through the provided features to apply each one
    for feature in features:
        feature_landmarks = None
        if feature == 'Lipstick':
            feature_landmarks = normalize_landmarks(ret_landmarks, height, width, upper_lip + lower_lip)
            mask = lip_mask(src, feature_landmarks, data["Lipstick"])
            all_masks = cv2.addWeighted(all_masks, 1.0, mask, 0.4, 0.0)

        elif feature == 'Blush On':
            left_cheek_landmarks = normalize_landmarks(ret_landmarks, height, width, left_cheeks)
            left_cheek_mask = blush_mask(src, left_cheek_landmarks,  data["Blush On"], 15)
            all_masks = cv2.addWeighted(all_masks, 1.0, left_cheek_mask, 0.3, 0.0)

            right_cheek_landmarks = normalize_landmarks(ret_landmarks, height, width, right_cheeks)
            right_cheek_mask = blush_mask(src, right_cheek_landmarks, data["Blush On"], 15)
            all_masks = cv2.addWeighted(all_masks, 1.0, right_cheek_mask, 0.3, 0.0)

        elif feature == 'Foundation':
            face_landmarks = normalize_landmarks(ret_landmarks, height, width, face_contour)
            left_eye_landmarks = normalize_landmarks(ret_landmarks, height, width, left_eye)
            right_eye_landmarks = normalize_landmarks(ret_landmarks, height, width, right_eye)
            lips_landmarks = normalize_landmarks(ret_landmarks, height, width, lips)

            mask = create_face_mask(src, face_landmarks, left_eye_landmarks, right_eye_landmarks, lips_landmarks)
            foundation_output = apply_foundation(src, mask, data["Foundation"], intensity=0.2)
            all_masks = cv2.addWeighted(all_masks, 1.0, foundation_output - src, 0.2, 0.0)

        elif feature == 'Contour':
            left_cheekbone_landmarks = normalize_landmarks(ret_landmarks, height, width, left_cheekbone)
            left_cheekbone_mask = contour_mask(src, left_cheekbone_landmarks, data["Contour"], 1)
            output = cv2.addWeighted(output, 1.0, left_cheekbone_mask, 0.3, 0.0)

            right_cheekbone_landmarks = normalize_landmarks(ret_landmarks, height, width, right_cheekbone)
            right_cheekbone_mask = contour_mask(src, right_cheekbone_landmarks, data["Contour"], 1)
            output = cv2.addWeighted(output, 1.0, right_cheekbone_mask, 0.3, 0.0)

            left_forehead_landmarks = normalize_landmarks(ret_landmarks, height, width, left_forehead)
            left_forehead_mask = contour_mask(src, left_forehead_landmarks, data["Contour"], 1)
            output = cv2.addWeighted(output, 1.0, left_forehead_mask, 0.3, 0.0)

            right_forehead_landmarks = normalize_landmarks(ret_landmarks, height, width, right_forehead)
            right_forehead_mask = contour_mask(src, right_forehead_landmarks, data["Contour"], 1)
            output = cv2.addWeighted(output, 1.0, right_forehead_mask, 0.3, 0.0)

    output = cv2.addWeighted(output, 1.0, all_masks, 0.8, 0.0)

    if show_landmarks and feature_landmarks is not None:
        plot_landmarks(src, feature_landmarks, True)

    # Encode the processed image to bytes
    _, result_bytes = cv2.imencode('.jpg', output)
    return result_bytes.tobytes()


def apply_makeup(src: np.ndarray, is_stream: bool, features: list, show_landmarks: bool = False):
    """
    Applies makeup effects based on the list of features: ['lips', 'blush', 'foundation'].
    This version continues running even if no face is detected.
    """
    ret_landmarks = detect_landmarks(src, is_stream)

    # Check if landmarks were detected; if not, return the original frame
    if ret_landmarks is None:
        print("No face detected")
        return src  # No face detected, so return the original image

    height, width, _ = src.shape

    output = src.copy()
    all_masks = np.zeros_like(src)  # A combined mask for all features

    # Loop through the provided features to apply each one
    for feature in features:
        feature_landmarks = None
        if feature == 'lips':
            feature_landmarks = normalize_landmarks(ret_landmarks, height, width, upper_lip + lower_lip)
            mask = lip_mask(src, feature_landmarks, [0, 0, 200])  # Lipstick color
            all_masks = cv2.addWeighted(all_masks, 1.0, mask, 0.4, 0.0)

        elif feature == 'blush':
            # Apply blush to the left cheek
            left_cheek_landmarks = normalize_landmarks(ret_landmarks, height, width, left_cheeks)
            left_cheek_mask = blush_mask(src, left_cheek_landmarks, [153, 0, 157], 15)  # Light pink color
            all_masks = cv2.addWeighted(all_masks, 1.0, left_cheek_mask, 0.3, 0.0)

            # Apply blush to the right cheek
            right_cheek_landmarks = normalize_landmarks(ret_landmarks, height, width, right_cheeks)
            right_cheek_mask = blush_mask(src, right_cheek_landmarks, [153, 0, 157], 15)  # Light pink color
            all_masks = cv2.addWeighted(all_masks, 1.0, right_cheek_mask, 0.3, 0.0)

        elif feature == 'foundation':
            face_landmarks = normalize_landmarks(ret_landmarks, height, width, face_contour)
            left_eye_landmarks = normalize_landmarks(ret_landmarks, height, width, left_eye)
            right_eye_landmarks = normalize_landmarks(ret_landmarks, height, width, right_eye)
            lips_landmarks = normalize_landmarks(ret_landmarks, height, width, lips)

            # Create a face mask that excludes eyes and lips
            mask = create_face_mask(src, face_landmarks, left_eye_landmarks, right_eye_landmarks, lips_landmarks)
            foundation_output = apply_foundation(src, mask, [224, 192, 160], intensity=0.3)
            all_masks = cv2.addWeighted(all_masks, 1.0, foundation_output - src, 0.2, 0.0)  # Blend foundation


        elif feature == 'contour':

            # Left cheekbone
            left_cheekbone_landmarks = normalize_landmarks(ret_landmarks, height, width, left_cheekbone)
            left_cheekbone_mask = contour_mask(src, left_cheekbone_landmarks, [200, 200, 200], 1)
            output = cv2.addWeighted(output, 1.0, left_cheekbone_mask, 0.3, 0.0)

            # Right cheekbone
            right_cheekbone_landmarks = normalize_landmarks(ret_landmarks, height, width, right_cheekbone)
            right_cheekbone_mask = contour_mask(src, right_cheekbone_landmarks, [200, 200, 200], 1)
            output = cv2.addWeighted(output, 1.0, right_cheekbone_mask, 0.3, 0.0)  # Left cheekbone

            left_forehead_landmarks = normalize_landmarks(ret_landmarks, height, width, left_forehead)
            left_forehead_mask = contour_mask(src, left_forehead_landmarks, [200, 200, 200], 1)
            output = cv2.addWeighted(output, 1.0, left_forehead_mask, 0.3, 0.0)


            right_forehead_landmarks = normalize_landmarks(ret_landmarks, height, width, right_forehead)
            right_forehead_mask = contour_mask(src, right_forehead_landmarks, [200, 200, 200], 1)
            output = cv2.addWeighted(output, 1.0, right_forehead_mask, 0.3, 0.0)

    # Apply the combined masks on the original image
    output = cv2.addWeighted(output, 1.0, all_masks, 0.8, 0.0)

    # Optionally, show landmarks for the last processed feature
    if show_landmarks and feature_landmarks is not None:
        plot_landmarks(src, feature_landmarks, True)

    return output


def contour_mask(src, landmarks, color, intensity):
    """
    Creates a smooth contour effect on the given regions.
    Arguments:
    - src: Source image.
    - landmarks: Landmarks outlining the contour area.
    - color: RGB color for contouring.
    - intensity: Float to adjust how strong the contour effect is (range 0.0 to 1.0).
    """
    # Create a blank mask
    mask = np.zeros_like(src, dtype=np.uint8)

    # Define the contour region as a polygon
    contour_points = np.array(landmarks, dtype=np.int32)
    cv2.fillPoly(mask, [contour_points], color)

    # Apply Gaussian blur to smooth the edges of the mask
    blurred_mask = cv2.GaussianBlur(mask, (25, 25), 0)

    # Blend the smoothed mask with the source image
    return cv2.addWeighted(src, 1.0, blurred_mask, intensity, 0.0)


def apply_feature(src: np.ndarray, feature: str, landmarks: list, normalize: bool = False,
                  show_landmarks: bool = False):
    """
    Performs similar to `apply_makeup` but needs the landmarks explicitly
    Specifically implemented to reduce the computation on the server
    """
    height, width, _ = src.shape
    if normalize:
        landmarks = normalize_landmarks(landmarks, height, width)
    if feature == 'lips':
        mask = lip_mask(src, landmarks, [153, 0, 157])
        output = cv2.addWeighted(src, 1.0, mask, 0.4, 0.0)
    elif feature == 'blush':
        mask = blush_mask(src, landmarks, [153, 0, 157], 50)
        output = cv2.addWeighted(src, 1.0, mask, 0.3, 0.0)
    else:  # Does not require any landmarks for skin masking -> Foundation
        skin_mask = mask_skin(src)
        output = np.where(src * skin_mask >= 1, gamma_correction(src, 1.75), src)
    if show_landmarks:  # Refrain from using this during an API Call
        plot_landmarks(src, landmarks, True)
    return output


def create_face_mask(src: np.ndarray, face_landmarks: np.ndarray, left_eye_landmarks: np.ndarray,
                     right_eye_landmarks: np.ndarray, lips_landmarks: np.ndarray):
    """
    Creates a binary mask based on the facial landmarks, excluding the eyes and mouth area.
    Arguments:
    - src: Source image.
    - face_landmarks: Normalized face landmarks (points outlining the face contour).
    - left_eye_landmarks: Normalized landmarks for the left eye.
    - right_eye_landmarks: Normalized landmarks for the right eye.
    - lips_landmarks: Normalized landmarks for the lips/mouth.
    """
    # Create an empty mask
    mask = np.zeros(src.shape[:2], dtype=np.uint8)

    # Convert face landmarks into an array of points for the face region
    face_points = np.array(face_landmarks, dtype=np.int32)
    left_eye_points = np.array(left_eye_landmarks, dtype=np.int32)
    right_eye_points = np.array(right_eye_landmarks, dtype=np.int32)
    lips_points = np.array(lips_landmarks, dtype=np.int32)

    # Fill the face mask with the face contour (polygon)
    cv2.fillPoly(mask, [face_points], 1)

    # Expand eye regions to cover more area (convex hull)
    left_eye_hull = cv2.convexHull(left_eye_points)
    right_eye_hull = cv2.convexHull(right_eye_points)

    # Fill in the eyes and lips as black areas to exclude them
    cv2.fillPoly(mask, [left_eye_hull], 0)
    cv2.fillPoly(mask, [right_eye_hull], 0)
    cv2.fillPoly(mask, [lips_points], 0)

    return mask


def apply_foundation(src: np.ndarray, face_mask: np.ndarray, foundation_color: list, intensity: float = 0.5):
    """
    Applies foundation to the face region based on the face mask.
    Arguments:
    - src: Source image.
    - face_mask: Binary mask identifying face region excluding eyes and lips.
    - foundation_color: RGB color for the foundation.
    - intensity: Float to adjust how strong the foundation effect is (range 0.0 to 1.0).
    """
    foundation_color_rgb = [foundation_color[2], foundation_color[1], foundation_color[0]]

    # Create a blank canvas for the foundation
    foundation_layer = np.zeros_like(src)
    foundation_layer[:] = foundation_color_rgb  # Apply the foundation color
    print(foundation_color_rgb)

    # # Blur the foundation layer to make it look more natural
    foundation_layer = cv2.GaussianBlur(foundation_layer, (15, 15), 10)

    # Blend the foundation with the original image using the face mask
    foundation_applied = cv2.addWeighted(src, 1.0 - intensity, foundation_layer, intensity, 0)

    # Expand the mask to 3 channels to match the source image shape
    face_mask_3channel = np.dstack([face_mask] * 3)  # Convert mask to 3-channel

    # Apply the foundation only to the face region using the mask
    output = np.where(face_mask_3channel == 1, foundation_applied, src)

    return output


def lip_mask(src: np.ndarray, points: np.ndarray, color: list):
    """
    Given a src image, points of lips and a desired color
    Returns a colored mask that can be added to the src
    """
    mask = np.zeros_like(src)  # Create a mask
    mask = cv2.fillPoly(mask, [points], color)  # Mask for the required facial feature
    # Blurring the region, so it looks natural
    # TODO: Get glossy finishes for lip colors, instead of blending in replace the region
    mask = cv2.GaussianBlur(mask, (7, 7), 5)
    return mask


def blush_mask(src: np.ndarray, points: np.ndarray, color: list, thickness: int):
    """
    Given a src image, points of the cheeks, desired color, and thickness,
    this function draws a blush effect in the form of lines instead of circles.

    Arguments:
    - src: Source image.
    - points: Landmarks representing cheek area.
    - color: RGB color for the blush.
    - thickness: Thickness of the blush line.

    Returns:
    - mask: Blush mask that can be added to the src.
    """
    # Create a blank mask
    mask = np.zeros_like(src)

    # Convert points to integer format for drawing
    points = points.astype(np.int32)

    # Draw lines between cheek points to create the blush effect
    # Loop over the points and draw lines
    for i in range(len(points) - 1):
        cv2.line(mask, tuple(points[i]), tuple(points[i + 1]), color, thickness)

    # Optionally, blur the lines to create a more natural look
    mask = cv2.GaussianBlur(mask, (15, 15), 10)

    return mask


def mask_skin(src: np.ndarray):
    """
    Given a source image of a person (face image)
    returns a mask that can be identified as the skin
    """
    lower = np.array([0, 133, 77], dtype='uint8')  # The lobound of skin color
    upper = np.array([255, 173, 127], dtype='uint8')  # Upper bound of skin color
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)  # Convert to YCR_CB
    skin_mask = cv2.inRange(dst, lower, upper)  # Get the skin
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)[..., np.newaxis]  # Dilate to fill in blobs

    if skin_mask.ndim != 3:
        skin_mask = np.expand_dims(skin_mask, axis=-1)
    return (skin_mask / 255).astype("uint8")  # A binary mask containing only 1s and 0s


def face_mask(src: np.ndarray, points: np.ndarray):
    """
    Given a list of face landmarks, return a closed polygon mask for the same
    """
    mask = np.zeros_like(src)
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
    return mask


def clicked_at(event, x, y, flags, params):
    """
    A useful callback that spits out the landmark index when clicked on a particular landmark
    Note: Very sensitive to location, should be clicked exactly on the pixel
    """
    # TODO: Add some atol to np.allclose
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at {x, y}")
        point = np.array([x, y])
        landmarks = params.get("landmarks", None)
        image = params.get("image", None)
        if landmarks is not None and image is not None:
            for idx, landmark in enumerate(landmarks):
                if np.allclose(landmark, point):
                    print(f"Landmark: {idx}")
                    break
            print("Found no landmark close to the click")


def vignette(src: np.ndarray, sigma: int):
    """
    Given a src image and a sigma, returns a vignette of the src
    """
    height, width, _ = src.shape
    kernel_x = cv2.getGaussianKernel(width, sigma)
    kernel_y = cv2.getGaussianKernel(height, sigma)

    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    blurred = cv2.convertScaleAbs(src.copy() * np.expand_dims(mask, axis=-1))
    return blurred


def face_bbox(src: np.ndarray, offset_x: int = 0, offset_y: int = 0):
    """
    Performs face detection on a src image, return bounding box coordinates with
    an optional offset applied to the coordinates
    """
    height, width, _ = src.shape
    with FaceDetection(model_selection=0) as detector:  # 0 -> dist <= 2mts from the camera
        results = detector.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
    results = results.detections[0].location_data
    x_min, y_min = results.relative_bounding_box.xmin, results.relative_bounding_box.ymin
    box_height, box_width = results.relative_bounding_box.height, results.relative_bounding_box.width
    x_min = int(width * x_min) - offset_x
    y_min = int(height * y_min) - offset_y
    box_height, box_width = int(height * box_height) + offset_y, int(width * box_width) + offset_x
    return (x_min, y_min), (box_height, box_width)


def gamma_correction(src: np.ndarray, gamma: float, coefficient: int = 1):
    """
    Performs gamma correction on a source image
    gamma > 1 => Darker Image
    gamma < 1 => Brighted Image
    """
    dst = src.copy()
    dst = dst / 255.  # Converted to float64
    dst = coefficient * np.power(dst, gamma)
    dst = (dst * 255).astype('uint8')
    return dst
