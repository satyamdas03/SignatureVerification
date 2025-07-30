# # main.py

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# import numpy as np
# import cv2
# from skimage.metrics import structural_similarity as ssim

# app = FastAPI(
#     title="Signature Verification API",
#     description="Compare a real 'ground truth' signature against a document's signature and return a match accuracy score.",
#     version="1.0.0"
# )

# def compute_signature_match_accuracy(
#     img_bytes1: bytes,
#     img_bytes2: bytes,
#     size: tuple[int, int] = (300, 150)
# ) -> float:
#     """
#     Compute match accuracy between two signature images using SSIM.
#     :param img_bytes1: Raw bytes of the reference (real) signature image.
#     :param img_bytes2: Raw bytes of the document signature image.
#     :param size: (width, height) to which both images will be resized.
#     :return: Match accuracy as a percentage [0.0 – 100.0].
#     """
#     # Decode images from bytes into grayscale
#     arr1 = np.frombuffer(img_bytes1, np.uint8)
#     arr2 = np.frombuffer(img_bytes2, np.uint8)
#     img1 = cv2.imdecode(arr1, cv2.IMREAD_GRAYSCALE)
#     img2 = cv2.imdecode(arr2, cv2.IMREAD_GRAYSCALE)

#     if img1 is None or img2 is None:
#         raise ValueError("One of the uploaded files is not a valid image.")

#     # Binarize (invert so signature strokes are white on black background)
#     _, thr1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV)
#     _, thr2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)

#     # Resize to the same dimensions
#     thr1 = cv2.resize(thr1, size, interpolation=cv2.INTER_AREA)
#     thr2 = cv2.resize(thr2, size, interpolation=cv2.INTER_AREA)

#     # Compute SSIM between the two binarized images
#     score, _ = ssim(thr1, thr2, full=True)
#     # Convert from [-1,1] range to [0,100] percentage
#     return float(round(score * 100, 2))


# @app.post(
#     "/verify-signature",
#     response_class=JSONResponse,
#     summary="Verify two signature images",
#     response_description="Returns a JSON object with `match_accuracy`."
# )
# async def verify_signature(
#     real_signature: UploadFile = File(..., description="The ground-truth signature image"),
#     document_signature: UploadFile = File(..., description="The signature image extracted from the document")
# ) -> JSONResponse:
#     """
#     Compare a real signature and a document signature.
#     Returns JSON: { "match_accuracy": 97.24 }
#     """
#     # Read raw bytes
#     real_bytes = await real_signature.read()
#     doc_bytes  = await document_signature.read()

#     try:
#         accuracy = compute_signature_match_accuracy(real_bytes, doc_bytes)
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     return JSONResponse(content={"match_accuracy": accuracy})









# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

app = FastAPI(
    title="Signature Verification API",
    description="Compute match accuracy between two signatures using enhanced SSIM pipeline.",
    version="2.1.0"  # Updated version
)

# Fixed output size for comparison
TARGET_SIZE = (300, 150)

def preprocess_signature(img_bytes: bytes) -> np.ndarray:
    """
    Enhanced preprocessing:
      1) Decode to grayscale
      2) CLAHE for contrast normalization
      3) Denoise with bilateral filter
      4) Find signature region via Otsu threshold
      5) Crop to signature bounding box
      6) Resize with aspect ratio preservation
      7) Pad to target size (centered)
    Returns grayscale image (0-255) of size TARGET_SIZE.
    """
    # 1) Decode
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image file")

    # 2) CLAHE (adaptive contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(img)

    # 3) Edge-preserving denoising
    den = cv2.bilateralFilter(cl, d=11, sigmaColor=100, sigmaSpace=100)

    # 4) Create mask for signature localization
    _, thresh = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    
    # 5) Crop to signature region or use full image
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = den[y:y+h, x:x+w]
    else:
        cropped = den  # Fallback if no contours
    
    # 6-7) Preserve aspect ratio while resizing
    h, w = cropped.shape
    if h == 0 or w == 0:  # Handle empty images
        return np.zeros(TARGET_SIZE[::-1], dtype=np.uint8)
    
    # Calculate scaling factor
    scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # High-quality downscaling
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Center-pad to target dimensions
    canvas = np.full(TARGET_SIZE[::-1], 255, dtype=np.uint8)  # White background
    x_offset = (TARGET_SIZE[0] - new_w) // 2
    y_offset = (TARGET_SIZE[1] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def compute_signature_match_accuracy(
    img_bytes1: bytes,
    img_bytes2: bytes
) -> float:
    """
    Computes SSIM on preprocessed grayscale images with:
    - Dynamic range adjustment
    - Optimized SSIM parameters
    """
    img1 = preprocess_signature(img_bytes1)
    img2 = preprocess_signature(img_bytes2)
    
    # Normalize images to 0-1 range for SSIM
    img1_norm = img1.astype(np.float64) / 255.0
    img2_norm = img2.astype(np.float64) / 255.0
    
    # SSIM with smaller window and higher weights
    win_size = min(7, min(img1.shape) // 2 * 2 - 1)  # Ensure odd and < image size
    win_size = max(3, win_size)  # Minimum window size
    
    score = ssim(
        img1_norm, 
        img2_norm,
        win_size=win_size,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )
    
    # Map to percentage scale
    return float(round(max(0, min(100, score * 100)), 2))  # Clamped 0-100

# REST endpoint remains unchanged
@app.post(
    "/verify-signature",
    response_class=JSONResponse,
    summary="Verify two signature images",
    response_description="Returns JSON: { match_accuracy: float }"
)
async def verify_signature(
    real_signature: UploadFile = File(..., description="Ground-truth signature image"),
    document_signature: UploadFile = File(..., description="Signature image from document")
) -> JSONResponse:
    real_bytes = await real_signature.read()
    doc_bytes  = await document_signature.read()

    try:
        accuracy = compute_signature_match_accuracy(real_bytes, doc_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content={ "match_accuracy": accuracy })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)