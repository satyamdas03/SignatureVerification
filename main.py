# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

app = FastAPI(
    title="Signature Verification API",
    description="Compare a real 'ground truth' signature against a document's signature and return a match accuracy score.",
    version="1.0.0"
)

def compute_signature_match_accuracy(
    img_bytes1: bytes,
    img_bytes2: bytes,
    size: tuple[int, int] = (300, 150)
) -> float:
    """
    Compute match accuracy between two signature images using SSIM.
    :param img_bytes1: Raw bytes of the reference (real) signature image.
    :param img_bytes2: Raw bytes of the document signature image.
    :param size: (width, height) to which both images will be resized.
    :return: Match accuracy as a percentage [0.0 – 100.0].
    """
    # Decode images from bytes into grayscale
    arr1 = np.frombuffer(img_bytes1, np.uint8)
    arr2 = np.frombuffer(img_bytes2, np.uint8)
    img1 = cv2.imdecode(arr1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(arr2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("One of the uploaded files is not a valid image.")

    # Binarize (invert so signature strokes are white on black background)
    _, thr1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV)
    _, thr2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)

    # Resize to the same dimensions
    thr1 = cv2.resize(thr1, size, interpolation=cv2.INTER_AREA)
    thr2 = cv2.resize(thr2, size, interpolation=cv2.INTER_AREA)

    # Compute SSIM between the two binarized images
    score, _ = ssim(thr1, thr2, full=True)
    # Convert from [-1,1] range to [0,100] percentage
    return float(round(score * 100, 2))


@app.post(
    "/verify-signature",
    response_class=JSONResponse,
    summary="Verify two signature images",
    response_description="Returns a JSON object with `match_accuracy`."
)
async def verify_signature(
    real_signature: UploadFile = File(..., description="The ground-truth signature image"),
    document_signature: UploadFile = File(..., description="The signature image extracted from the document")
) -> JSONResponse:
    """
    Compare a real signature and a document signature.
    Returns JSON: { "match_accuracy": 97.24 }
    """
    # Read raw bytes
    real_bytes = await real_signature.read()
    doc_bytes  = await document_signature.read()

    try:
        accuracy = compute_signature_match_accuracy(real_bytes, doc_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content={"match_accuracy": accuracy})

















# # main.py (TRYING NEW METHOD)

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# import numpy as np
# import cv2
# from skimage.metrics import structural_similarity as ssim

# app = FastAPI(version="1.4.0")

# # --- Configurable Weights & Params ---
# WEIGHTS = {'ssim': 0.5, 'sift': 0.3, 'hu': 0.2}
# THRESH_BLOCK = 15      # for adaptive threshold
# THRESH_C = 5           # for adaptive threshold
# TOPHAT_KERNEL = (15,15)
# TARGET_SIZE = (300, 150)

# def morphological_enhance(img: np.ndarray) -> np.ndarray:
#     """
#     Apply top-hat to highlight strokes, and black-hat to remove background.
#     """
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, TOPHAT_KERNEL)
#     tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)  # white-hat :contentReference[oaicite:6]{index=6}
#     blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)  # suppress dark noise :contentReference[oaicite:7]{index=7}
#     enhanced = cv2.add(img, tophat)
#     enhanced = cv2.subtract(enhanced, blackhat)
#     return enhanced

# def decode_crop_deskew(img_bytes: bytes) -> np.ndarray:
#     """
#     Decode to grayscale, crop to ink ROI via Otsu, deskew by moments.
#     """
#     arr = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError("Invalid image")  # ensure valid decode

#     # Crop by Otsu for ROI
#     _, bin0 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     pts = cv2.findNonZero(bin0)
#     if pts is not None:
#         x, y, w, h = cv2.boundingRect(pts)
#         img = img[y:y+h, x:x+w]

#     # Deskew via image moments
#     m = cv2.moments(img)
#     if abs(m['mu02']) > 1e-2:
#         skew = m['mu11'] / m['mu02']  # baseline skew :contentReference[oaicite:8]{index=8}
#         M = np.array([[1, skew, -0.5 * img.shape[0] * skew],
#                       [0,    1,                           0]], dtype=np.float32)
#         img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
#                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
#     return img

# def translate_register(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
#     """
#     Align mov to ref via centroid translation only.
#     """
#     # create binary masks for centroids
#     _, m1 = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#     _, m2 = cv2.threshold(mov, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#     pts1 = np.column_stack(np.where(m1>0))
#     pts2 = np.column_stack(np.where(m2>0))
#     if pts1.size and pts2.size:
#         y1, x1 = pts1.mean(axis=0)
#         y2, x2 = pts2.mean(axis=0)
#         dy, dx = y1 - y2, x1 - x2
#         M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
#         mov = cv2.warpAffine(mov, M, (mov.shape[1], mov.shape[0]),
#                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
#     return mov  # simple translation alignment :contentReference[oaicite:9]{index=9}

# def binarize_adaptive(img: np.ndarray) -> np.ndarray:
#     """
#     Adaptive Gaussian threshold to robustly segment strokes.
#     """
#     return cv2.adaptiveThreshold(img, 255,
#                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                  cv2.THRESH_BINARY_INV,
#                                  THRESH_BLOCK, THRESH_C)  # robust under uneven lighting :contentReference[oaicite:10]{index=10}

# def compute_ssim_score(m1: np.ndarray, m2: np.ndarray) -> float:
#     """
#     Compute SSIM on resized masks.
#     """
#     r1 = cv2.resize(m1, TARGET_SIZE, interpolation=cv2.INTER_AREA)
#     r2 = cv2.resize(m2, TARGET_SIZE, interpolation=cv2.INTER_AREA)
#     score, _ = ssim(r1, r2, full=True)
#     return score

# def compute_sift_ratio(m1: np.ndarray, m2: np.ndarray) -> float:
#     """
#     Extract SIFT features, match with Lowe's ratio test, return normalized ratio.
#     """
#     sift = cv2.SIFT_create()  # SIFT detector :contentReference[oaicite:11]{index=11}
#     kp1, des1 = sift.detectAndCompute(m1, None)
#     kp2, des2 = sift.detectAndCompute(m2, None)
#     if des1 is None or des2 is None or not kp1 or not kp2:
#         return 0.0
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#     good = [m for m,n in matches if m.distance < 0.75 * n.distance]
#     ratio = len(good) / max(len(kp1), len(kp2))
#     return min(ratio, 1.0)

# def compute_hu_score(m1: np.ndarray, m2: np.ndarray) -> float:
#     """
#     Compute seven Hu moments on each mask and derive similarity.
#     """
#     hu1 = cv2.HuMoments(cv2.moments(m1)).flatten()
#     hu2 = cv2.HuMoments(cv2.moments(m2)).flatten()
#     # log-scale diffs to handle wide dynamic range of moments :contentReference[oaicite:12]{index=12}
#     d = np.sum(np.abs(np.log(np.abs(hu1)+1e-9) - np.log(np.abs(hu2)+1e-9)))
#     score = 1.0 / (1.0 + d)  # maps [0,∞) → (0,1]
#     return float(min(max(score, 0.0), 1.0))

# @app.post("/verify-signature", response_class=JSONResponse)
# async def verify_signature(
#     real_sig: UploadFile = File(..., description="Ground-truth signature"),
#     doc_sig:  UploadFile = File(..., description="Document signature")
# ):
#     b1 = await real_sig.read()
#     b2 = await doc_sig.read()
#     try:
#         # 1) Decode, crop, deskew
#         gt  = decode_crop_deskew(b1)
#         doc = decode_crop_deskew(b2)

#         # 2) Morphological enhancement
#         gt_en  = morphological_enhance(gt)
#         doc_en = morphological_enhance(doc)

#         # 3) Registration (translation-only)
#         doc_reg = translate_register(gt_en, doc_en)

#         # 4) Binarize adaptively
#         m1 = binarize_adaptive(gt_en)
#         m2 = binarize_adaptive(doc_reg)

#         # 5) Compute metrics
#         ssim_score = compute_ssim_score(m1, m2)
#         sift_ratio  = compute_sift_ratio(m1, m2)
#         hu_score    = compute_hu_score(m1, m2)

#         # 6) Weighted fusion
#         final = (WEIGHTS['ssim'] * ssim_score +
#                  WEIGHTS['sift'] * sift_ratio +
#                  WEIGHTS['hu']   * hu_score)
#         accuracy = float(round(final * 100, 2))
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     return JSONResponse(content={"match_accuracy": accuracy})

