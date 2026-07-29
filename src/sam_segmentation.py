import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from src.config import device

def load_sam(checkpoint_path):
    """load SAM"""
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

def generate_mask_with_gui(image_path, predictor, save_path=None):
    """
    GUI-based point selection for SAM mask generation.
    User clicks 3 points on the image (Tip, Middle, End) and SAM generates a mask.
    """
    # Load image and set it for the predictor
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    clicked_points = []
    display_image = image_bgr.copy()
    window_name = "SAM - Click 3 points (Tip -> Middle -> End)"

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_points, display_image
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append([x, y])
            print(f"Point {len(clicked_points)}: ({x}, {y})")
            
            # Draw the clicked point on the image
            cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_image, str(len(clicked_points)), 
                       (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            
            cv2.imshow(window_name, display_image)
            
            if len(clicked_points) >= 3:
                cv2.putText(display_image, "3 points selected! Press ENTER to confirm", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 0), 2)
                cv2.imshow(window_name, display_image)

    # Set up the OpenCV window and mouse callback
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 800)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Display the image and wait for user input
    cv2.imshow(window_name, display_image)
    print("Click 3 points on the image (Tip -> Middle -> End). Press ENTER to confirm, ESC to cancel.")

    # Wait for user to finish clicking points and press ENTER or ESC
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            break
        elif key == 27:  # ESC
            print("Selection cancelled by user.")
            cv2.destroyAllWindows()
            return None, None

    cv2.destroyAllWindows()

    if len(clicked_points) < 3:
        print(f"Only {len(clicked_points)} points selected. Need 3 points.")
        return None, None

    # Generate mask using the clicked points
    points = np.array(clicked_points)
    point_labels = np.ones(len(points), dtype=int)
    
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=point_labels,
        multimask_output=True
    )
    best_mask = masks[np.argmax(scores)]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, (best_mask * 255).astype(np.uint8))
        print(f"Mask saved to: {save_path}")

    return best_mask, points

# this function has the same function as generate_mask_with_gui (just no klicking process). This function will be used in the run_demo.py 
def generate_mask_with_points(img_bgr, predictor, points_override=None, save_path=None):
    """
    Generate SAM mask with pre-defined points (no GUI). Used for demo.
    Args:
        img_bgr: image in BGR format (numpy array)
        predictor: SAM predictor
        points_override: list of [x, y] points
        save_path: optional path to save mask
    Returns:
        best_mask, points
    """
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    if points_override is None:
        raise ValueError("points_override must be provided")
    
    points = np.array(points_override)
    point_labels = np.ones(len(points), dtype=int)

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=point_labels,
        multimask_output=True
    )
    best_mask = masks[np.argmax(scores)]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, (best_mask * 255).astype(np.uint8))
        print(f"Mask saved to: {save_path}")

    return best_mask, points

