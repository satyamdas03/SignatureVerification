# ##BETTER_MATCHING_ALGO (working well for similar looking signatures)
# import os, io, tempfile, shutil
# import numpy as np
# import cv2
# import fitz  # PyMuPDF
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# import docx2txt
# from skimage.feature import hog, local_binary_pattern
# from skimage.metrics import structural_similarity as ssim
# from scipy.spatial.distance import cosine

# app = FastAPI(version="3.1.0")

# # -------------------------------------------------------------------
# # 1) Document → Signature Extraction (Unchanged)
# # -------------------------------------------------------------------

# def extract_signature_from_pdf(pdf_bytes: bytes) -> np.ndarray:
#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#     page = doc[-1]
#     images = page.get_images(full=True)
#     if not images:
#         doc.close()
#         raise ValueError("No images found in PDF")

#     best_mask = None
#     best_ratio = 0.0

#     for img_meta in images:
#         xref = img_meta[0]
#         pix = fitz.Pixmap(doc, xref)
#         if pix.n > 4:
#             pix = fitz.Pixmap(pix, 0)
#         arr = np.frombuffer(pix.samples, np.uint8)
#         channels = pix.n
#         arr = arr.reshape((pix.height, pix.width, channels))
#         pix = None

#         if channels >= 3:
#             gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
#         else:
#             gray = arr

#         _, mask = cv2.threshold(
#             gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#         )

#         ratio = cv2.countNonZero(mask) / (mask.size)
#         if ratio > best_ratio:
#             best_ratio = ratio
#             best_mask = mask

#     doc.close()
#     if best_mask is None:
#         raise ValueError("Could not extract signature from PDF images")

#     return best_mask

# def extract_signature_from_docx(docx_bytes: bytes) -> np.ndarray:
#     tmp_dir = tempfile.mkdtemp()
#     docx_path = os.path.join(tmp_dir, "input.docx")
#     with open(docx_path, "wb") as f:
#         f.write(docx_bytes)

#     img_dir = os.path.join(tmp_dir, "imgs")
#     os.makedirs(img_dir, exist_ok=True)
#     docx2txt.process(docx_path, img_dir)

#     candidates = []
#     for fn in os.listdir(img_dir):
#         if fn.lower().endswith((".png", ".jpg", ".jpeg")):
#             full = os.path.join(img_dir, fn)
#             img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
#             if img is not None:
#                 candidates.append((full, img.shape[0]*img.shape[1], img))

#     shutil.rmtree(tmp_dir)

#     if not candidates:
#         raise ValueError("No images found in DOCX")

#     _, _, best_img = max(candidates, key=lambda x: x[1])

#     _, mask = cv2.threshold(
#         best_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )
#     return mask

# # -------------------------------------------------------------------
# # 2) Enhanced Preprocessing & Comparison
# # -------------------------------------------------------------------

# def standardize_signature(image: np.ndarray, target_size: tuple = (500, 500)) -> np.ndarray:
#     """Resize signature while maintaining aspect ratio with padding"""
#     if image is None:
#         return None
        
#     h, w = image.shape
#     ratio = min(target_size[0] / h, target_size[1] / w)
#     new_h, new_w = int(h * ratio), int(w * ratio)
    
#     # Preserve binary nature with nearest-neighbor interpolation
#     resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
#     # Pad to target size with background (black)
#     top = (target_size[0] - new_h) // 2
#     bottom = target_size[0] - new_h - top
#     left = (target_size[1] - new_w) // 2
#     right = target_size[1] - new_w - left
#     padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
#                                cv2.BORDER_CONSTANT, value=0)
#     return padded

# def clean_signature(mask: np.ndarray) -> np.ndarray:
#     """Morphological cleaning without resizing"""
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# def compute_match_accuracy(sig1: np.ndarray, sig2: np.ndarray) -> float:
#     """Advanced signature matching using HOG, LBP, and SSIM features"""
#     # Standardize both signatures to 500x500
#     std1 = standardize_signature(sig1, (500, 500))
#     std2 = standardize_signature(sig2, (500, 500))
    
#     # Convert to uint8 (0-255) if needed
#     std1 = std1.astype(np.uint8)
#     std2 = std2.astype(np.uint8)
    
#     # HOG Feature Similarity
#     fd1, _ = hog(std1, orientations=8, pixels_per_cell=(16, 16),
#                  cells_per_block=(1, 1), visualize=True, channel_axis=None)
#     fd2, _ = hog(std2, orientations=8, pixels_per_cell=(16, 16),
#                  cells_per_block=(1, 1), visualize=True, channel_axis=None)
    
#     hog_sim = 1 - cosine(fd1, fd2) if (np.linalg.norm(fd1) > 0 and np.linalg.norm(fd2) > 0) else 0.0
#     hog_sim = max(0.0, min(1.0, hog_sim))
    
#     # LBP Feature Similarity
#     radius = 3
#     n_points = 8 * radius
#     lbp1 = local_binary_pattern(std1, n_points, radius, method='uniform')
#     lbp2 = local_binary_pattern(std2, n_points, radius, method='uniform')
    
#     n_bins = int(lbp1.max() + 1)
#     hist1, _ = np.histogram(lbp1.ravel(), bins=n_bins, range=(0, n_bins), density=True)
#     hist2, _ = np.histogram(lbp2.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
#     lbp_sim = np.sum(np.sqrt(hist1 * hist2))
#     lbp_sim = max(0.0, min(1.0, lbp_sim))
    
#     # SSIM Structural Similarity
#     ssim_score = ssim(std1, std2, data_range=255)
#     ssim_score = max(0.0, min(1.0, ssim_score))
    
#     # Weighted combination (adjust weights as needed)
#     weights = {'hog': 0.4, 'lbp': 0.4, 'ssim': 0.2}
#     combined_score = (weights['hog'] * hog_sim +
#                      weights['lbp'] * lbp_sim +
#                      weights['ssim'] * ssim_score)
    
#     # Convert to percentage
#     return max(0.0, min(100.0, combined_score * 100))

# # -------------------------------------------------------------------
# # 3) FastAPI Endpoint (Modified)
# # -------------------------------------------------------------------

# @app.post("/verify-signature", response_class=JSONResponse)
# async def verify_signature(
#     doc1: UploadFile = File(..., description="First document (PDF or DOCX)"),
#     doc2: UploadFile = File(..., description="Second document (PDF or DOCX)")
# ):
#     b1, b2 = await doc1.read(), await doc2.read()

#     try:
#         # Extract signatures
#         if doc1.filename.lower().endswith(".pdf"):
#             sig1 = extract_signature_from_pdf(b1)
#         elif doc1.filename.lower().endswith(".docx"):
#             sig1 = extract_signature_from_docx(b1)
#         else:
#             raise HTTPException(400, f"Unsupported type for doc1: {doc1.filename}")

#         if doc2.filename.lower().endswith(".pdf"):
#             sig2 = extract_signature_from_pdf(b2)
#         elif doc2.filename.lower().endswith(".docx"):
#             sig2 = extract_signature_from_docx(b2)
#         else:
#             raise HTTPException(400, f"Unsupported type for doc2: {doc2.filename}")

#         # Clean signatures (morphological operations)
#         cleaned_sig1 = clean_signature(sig1)
#         cleaned_sig2 = clean_signature(sig2)
        
#         # Calculate match accuracy with advanced method
#         accuracy = compute_match_accuracy(cleaned_sig1, cleaned_sig2)

#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     return JSONResponse({"match_accuracy": round(accuracy, 2)})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



















## BETTER ALGO?
import io, os, math, tempfile, shutil
import numpy as np
import cv2
import fitz  # PyMuPDF
import docx2txt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Tuple, Optional, List, Dict
from skimage.morphology import skeletonize

app = FastAPI(version="4.0.0")


# -----------------------------
# Utils
# -----------------------------
def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        raise ValueError("Empty image")
    if len(img_bgr.shape) == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        31, 15
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return bw


def _deskew_to_horizontal(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return mask
    pts = np.column_stack((xs, ys)).astype(np.float32)
    # PCA angle
    mean = np.mean(pts, axis=0)
    pts_c = pts - mean
    cov = np.cov(pts_c.T)
    w, v = np.linalg.eig(cov)
    major = v[:, np.argmax(w)]
    angle = math.degrees(math.atan2(major[1], major[0]))
    # rotate to near-horizontal within [-45,45]
    rot_angle = angle if abs(angle) <= 60 else (angle - 90 if angle > 0 else angle + 90)
    h, w_ = mask.shape
    M = cv2.getRotationMatrix2D((w_//2, h//2), rot_angle, 1.0)
    rotated = cv2.warpAffine(mask, M, (w_, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated


def _tight_crop(mask: np.ndarray, pad: int = 8) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = mask.shape
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    return mask[y1:y2+1, x1:x2+1]


def _resize_pad(mask: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    h, w = mask.shape
    scale = min(size[0] / h, size[1] / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    top = (size[0] - nh) // 2
    bottom = size[0] - nh - top
    left = (size[1] - nw) // 2
    right = size[1] - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


# -----------------------------
# Signature detection per image
# -----------------------------
def _score_component(comp_mask: np.ndarray, page_h: int, page_w: int, bbox: Tuple[int,int,int,int]) -> float:
    # bbox: x, y, w, h
    x, y, w, h = bbox
    area = int(comp_mask.sum() // 255)
    box_area = w * h
    page_area = page_h * page_w

    if box_area <= 0:
        return 0.0

    extent = area / (box_area + 1e-6)           # ink density inside its box
    if area < 0.0002 * page_area or area > 0.05 * page_area:
        return 0.0

    ar = w / (h + 1e-6)                         # aspect ratio
    if not (1.6 <= ar <= 14):
        return 0.0

    # contour complexity (more curvy than blocky)
    contours, _ = cv2.findContours((comp_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    perim = cv2.arcLength(c, True) + 1e-6
    comp_area = cv2.contourArea(c) + 1e-6
    circularity = (4 * math.pi * comp_area) / (perim * perim)
    complexity = 1 - circularity                 # higher is more signature-like

    # skeleton length / area (signatures are sparse strokes)
    sk = skeletonize((comp_mask > 0).astype(bool)).astype(np.uint8) * 255
    sk_len = int(sk.sum() // 255)
    stroke_ratio = sk_len / (area + 1e-6)

    # small bottom bias (not required, just a nudge)
    bottom_bias = (y + h) / page_h

    # aggregate score
    score = (
        0.35 * extent +
        0.30 * complexity +
        0.25 * min(1.5, stroke_ratio) / 1.5 +
        0.10 * bottom_bias
    )
    return float(score)


def detect_signature_from_page(gray: np.ndarray) -> Optional[np.ndarray]:
    H, W = gray.shape
    bw = _adaptive_binarize(gray)

    # remove horizontal/vertical rules & large blocks quickly
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (W//20 or 1, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, H//20 or 1)))
    bw = cv2.subtract(bw, horiz)
    bw = cv2.subtract(bw, vert)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return None

    best = None
    best_score = 0.0
    for lab in range(1, num):
        x, y, w, h, area = stats[lab]
        if area < 25:
            continue
        comp = (labels == lab).astype(np.uint8) * 255
        comp_roi = comp[y:y+h, x:x+w]
        score = _score_component(comp_roi, H, W, (x, y, w, h))
        if score > best_score:
            best_score = score
            best = comp_roi

    if best is None:
        return None

    best = _deskew_to_horizontal(best)
    best = _tight_crop(best, pad=8)
    best = cv2.morphologyEx(best, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return best if best.sum() > 0 else None


# -----------------------------
# File-type loaders
# -----------------------------
def extract_signature_from_pdf(pdf_bytes: bytes) -> np.ndarray:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        best_mask, best_page_score = None, -1.0
        for page in doc:
            # render at ~300 DPI
            zoom = 300.0 / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = detect_signature_from_page(gray)
            if mask is not None:
                # prefer more “signature-like” (use our component scoring again)
                score = _score_component(mask, mask.shape[0], mask.shape[1], (0, 0, mask.shape[1], mask.shape[0]))
                if score > best_page_score:
                    best_page_score = score
                    best_mask = mask
        if best_mask is None:
            raise ValueError("Signature not found in PDF")
        return best_mask
    finally:
        doc.close()


def extract_signature_from_docx(docx_bytes: bytes) -> np.ndarray:
    tmp_dir = tempfile.mkdtemp()
    try:
        docx_path = os.path.join(tmp_dir, "input.docx")
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)

        img_dir = os.path.join(tmp_dir, "imgs")
        os.makedirs(img_dir, exist_ok=True)
        docx2txt.process(docx_path, img_dir)

        best, best_score = None, -1.0
        for fn in os.listdir(img_dir):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = cv2.imread(os.path.join(img_dir, fn))
            if img is None:
                continue
            gray = _to_gray(img)
            mask = detect_signature_from_page(gray)
            if mask is None:
                continue
            score = _score_component(mask, mask.shape[0], mask.shape[1], (0, 0, mask.shape[1], mask.shape[0]))
            if score > best_score:
                best_score, best = score, mask

        if best is None:
            raise ValueError("Signature not found in DOCX (no suitable images)")
        return best
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_signature_from_image(img_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    gray = _to_gray(img)
    mask = detect_signature_from_page(gray)
    if mask is None:
        raise ValueError("Signature not found in image")
    return mask


# -----------------------------
# Matching (robust)
# -----------------------------
def _prep_for_match(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = _deskew_to_horizontal(mask)
    mask = _tight_crop(mask, pad=8)
    mask = _resize_pad(mask, (256, 256))
    skel = skeletonize((mask > 0).astype(bool)).astype(np.uint8) * 255
    edges = cv2.Canny(mask, 50, 120)
    return mask, skel, edges


def _hu_similarity(maskA: np.ndarray, maskB: np.ndarray) -> float:
    m1 = cv2.moments((maskA > 0).astype(np.uint8))
    m2 = cv2.moments((maskB > 0).astype(np.uint8))
    if abs(m1["m00"]) < 1e-6 or abs(m2["m00"]) < 1e-6:
        return 0.0
    h1 = cv2.HuMoments(m1).flatten()
    h2 = cv2.HuMoments(m2).flatten()
    # log transform
    h1 = np.sign(h1) * np.log1p(np.abs(h1))
    h2 = np.sign(h2) * np.log1p(np.abs(h2))
    dist = np.linalg.norm(h1 - h2)
    sim = math.exp(-0.75 * dist)  # 0..1
    return float(sim)


def _orb_similarity(edgesA: np.ndarray, edgesB: np.ndarray) -> float:
    orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, edgeThreshold=15, patchSize=31)
    kp1, des1 = orb.detectAndCompute(edgesA, None)
    kp2, des2 = orb.detectAndCompute(edgesB, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    base = max(8, min(len(kp1), len(kp2)))
    return float(min(1.0, len(good) / base))


def _chamfer_similarity(edgesA: np.ndarray, edgesB: np.ndarray) -> float:
    dtA = cv2.distanceTransform(255 - edgesA, cv2.DIST_L2, 3)
    dtB = cv2.distanceTransform(255 - edgesB, cv2.DIST_L2, 3)
    a_on_b = dtB[edgesA > 0].mean() if np.any(edgesA > 0) else 1e3
    b_on_a = dtA[edgesB > 0].mean() if np.any(edgesB > 0) else 1e3
    d = (a_on_b + b_on_a) / 2.0
    # normalize by image diagonal
    diag = math.hypot(*edgesA.shape)
    sim = max(0.0, 1.0 - (d / (0.06 * diag + 1e-6)))  # 0 when far, ~1 when edges overlap
    return float(sim)


def compute_match_accuracy(sig1: np.ndarray, sig2: np.ndarray) -> Tuple[float, Dict[str, float]]:
    m1, sk1, e1 = _prep_for_match(sig1)
    m2, sk2, e2 = _prep_for_match(sig2)

    hu = _hu_similarity(m1, m2)
    orb = _orb_similarity(e1, e2)
    chamfer = _chamfer_similarity(e1, e2)

    # light structural check (SSIM on masks)
    ssim = cv2.quality.QualitySSIM_compute(m1, m2)[0][0] if hasattr(cv2, "quality") else 0.0
    ssim = float(np.clip(ssim, 0, 1))

    # combine with gating (prevents high scores for random scribbles)
    base = 0.50 * orb + 0.25 * hu + 0.15 * chamfer + 0.10 * ssim
    if orb < 0.15 and hu < 0.35:
        base *= 0.4
    accuracy = float(np.clip(base * 100.0, 0.0, 100.0))

    breakdown = {"orb": round(orb, 3), "hu": round(hu, 3), "chamfer": round(chamfer, 3), "ssim": round(ssim, 3)}
    return accuracy, breakdown


# -----------------------------
# Router
# -----------------------------
def extract_signature_any(filename: str, data: bytes) -> np.ndarray:
    ext = os.path.splitext(filename.lower())[-1]
    if ext == ".pdf":
        return extract_signature_from_pdf(data)
    elif ext == ".docx":
        return extract_signature_from_docx(data)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        return extract_signature_from_image(data)
    else:
        raise HTTPException(400, f"Unsupported file type: {ext}")


@app.post("/verify-signature", response_class=JSONResponse)
async def verify_signature(
    doc1: UploadFile = File(..., description="PDF/DOCX/PNG/JPG"),
    doc2: UploadFile = File(..., description="PDF/DOCX/PNG/JPG"),
    return_breakdown: bool = False
):
    b1, b2 = await doc1.read(), await doc2.read()
    try:
        sig1 = extract_signature_any(doc1.filename, b1)
        sig2 = extract_signature_any(doc2.filename, b2)

        acc, breakdown = compute_match_accuracy(sig1, sig2)
        result = {"match_accuracy": round(acc, 2)}
        if return_breakdown:
            result["breakdown"] = breakdown
        return JSONResponse(result)

    except ValueError as e:
        raise HTTPException(400, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
