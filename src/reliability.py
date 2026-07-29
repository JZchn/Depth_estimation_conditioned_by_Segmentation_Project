import cv2
import numpy as np

def compute_reliability_and_features(img_bgr, mask_path):
    """
    Compute reliability map and four feature maps from input image.
    Input: BGR image, SAM mask path
    Returns: reliability_map, float_mask, features_dict (4 features)
    """
    # Image preprocessing
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Load and blur mask
    mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
    mask_blur = cv2.GaussianBlur(mask, (15, 15), 5)

    # 1. Specular highlights
    h, s, v = cv2.split(img_hsv)
    specular = ((v > 220) & (s < 50)).astype(np.float32)

    # 2. Shading
    shading = cv2.GaussianBlur(img_gray, (0, 0), 15)
    shading = (shading - shading.min()) / (shading.max() + 1e-8)

    # 3. Texture / Local Contrast
    local_std = cv2.blur(img_gray**2, (7, 7)) - cv2.blur(img_gray, (7, 7))**2
    local_std = np.sqrt(np.maximum(local_std, 0))
    local_std = local_std / (local_std.max() + 1e-8)

    # 4. Gradient / Edge
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    grad = grad / (grad.max() + 1e-8)

    # Reliability weights (hyperparameters)
    w_shading = 0.5
    w_texture = 0.3
    w_edge = 0.2
    w_spec = 0.8

    # Compute base reliability R
    R = (w_shading * shading + w_texture * local_std + w_edge * grad) * (1 - mask_blur)
    R = R * (1 - w_spec * specular)
    R = np.clip(R, 0, 1)

    features = {
        "Shading": shading,
        "Texture": local_std,
        "Edge": grad,
        "Specular": specular
    }

    return R.astype(np.float32), mask.astype(np.float32), features

def apply_reliability_to_image(img_bgr, reliability):
    """Apply reliability-weighted multiplication to the image (brightness suppression)."""
    img_f = img_bgr.astype(np.float32) / 255.0
    img_w = img_f * reliability[..., None]
    img_w = np.clip(img_w * 255.0, 0, 255).astype(np.uint8)
    return img_w