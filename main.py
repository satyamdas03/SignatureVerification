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












##BETTER_MATCHING_ALGO (working well for similar looking signatures)
import os, io, tempfile, shutil
import numpy as np
import cv2
import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import docx2txt
from skimage.feature import hog, local_binary_pattern
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine

app = FastAPI(version="3.1.0")

# -------------------------------------------------------------------
# 1) Document → Signature Extraction (Unchanged)
# -------------------------------------------------------------------

def extract_signature_from_pdf(pdf_bytes: bytes) -> np.ndarray:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[-1]
    images = page.get_images(full=True)
    if not images:
        doc.close()
        raise ValueError("No images found in PDF")

    best_mask = None
    best_ratio = 0.0

    for img_meta in images:
        xref = img_meta[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n > 4:
            pix = fitz.Pixmap(pix, 0)
        arr = np.frombuffer(pix.samples, np.uint8)
        channels = pix.n
        arr = arr.reshape((pix.height, pix.width, channels))
        pix = None

        if channels >= 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        ratio = cv2.countNonZero(mask) / (mask.size)
        if ratio > best_ratio:
            best_ratio = ratio
            best_mask = mask

    doc.close()
    if best_mask is None:
        raise ValueError("Could not extract signature from PDF images")

    return best_mask

def extract_signature_from_docx(docx_bytes: bytes) -> np.ndarray:
    tmp_dir = tempfile.mkdtemp()
    docx_path = os.path.join(tmp_dir, "input.docx")
    with open(docx_path, "wb") as f:
        f.write(docx_bytes)

    img_dir = os.path.join(tmp_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    docx2txt.process(docx_path, img_dir)

    candidates = []
    for fn in os.listdir(img_dir):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            full = os.path.join(img_dir, fn)
            img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                candidates.append((full, img.shape[0]*img.shape[1], img))

    shutil.rmtree(tmp_dir)

    if not candidates:
        raise ValueError("No images found in DOCX")

    _, _, best_img = max(candidates, key=lambda x: x[1])

    _, mask = cv2.threshold(
        best_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return mask

# -------------------------------------------------------------------
# 2) Enhanced Preprocessing & Comparison
# -------------------------------------------------------------------

def standardize_signature(image: np.ndarray, target_size: tuple = (500, 500)) -> np.ndarray:
    """Resize signature while maintaining aspect ratio with padding"""
    if image is None:
        return None
        
    h, w = image.shape
    ratio = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # Preserve binary nature with nearest-neighbor interpolation
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Pad to target size with background (black)
    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=0)
    return padded

def clean_signature(mask: np.ndarray) -> np.ndarray:
    """Morphological cleaning without resizing"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def compute_match_accuracy(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Advanced signature matching using HOG, LBP, and SSIM features"""
    # Standardize both signatures to 500x500
    std1 = standardize_signature(sig1, (500, 500))
    std2 = standardize_signature(sig2, (500, 500))
    
    # Convert to uint8 (0-255) if needed
    std1 = std1.astype(np.uint8)
    std2 = std2.astype(np.uint8)
    
    # HOG Feature Similarity
    fd1, _ = hog(std1, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=True, channel_axis=None)
    fd2, _ = hog(std2, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=True, channel_axis=None)
    
    hog_sim = 1 - cosine(fd1, fd2) if (np.linalg.norm(fd1) > 0 and np.linalg.norm(fd2) > 0) else 0.0
    hog_sim = max(0.0, min(1.0, hog_sim))
    
    # LBP Feature Similarity
    radius = 3
    n_points = 8 * radius
    lbp1 = local_binary_pattern(std1, n_points, radius, method='uniform')
    lbp2 = local_binary_pattern(std2, n_points, radius, method='uniform')
    
    n_bins = int(lbp1.max() + 1)
    hist1, _ = np.histogram(lbp1.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist2, _ = np.histogram(lbp2.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    lbp_sim = np.sum(np.sqrt(hist1 * hist2))
    lbp_sim = max(0.0, min(1.0, lbp_sim))
    
    # SSIM Structural Similarity
    ssim_score = ssim(std1, std2, data_range=255)
    ssim_score = max(0.0, min(1.0, ssim_score))
    
    # Weighted combination (adjust weights as needed)
    weights = {'hog': 0.4, 'lbp': 0.4, 'ssim': 0.2}
    combined_score = (weights['hog'] * hog_sim +
                     weights['lbp'] * lbp_sim +
                     weights['ssim'] * ssim_score)
    
    # Convert to percentage
    return max(0.0, min(100.0, combined_score * 100))

# -------------------------------------------------------------------
# 3) FastAPI Endpoint (Modified)
# -------------------------------------------------------------------

@app.post("/verify-signature", response_class=JSONResponse)
async def verify_signature(
    doc1: UploadFile = File(..., description="First document (PDF or DOCX)"),
    doc2: UploadFile = File(..., description="Second document (PDF or DOCX)")
):
    b1, b2 = await doc1.read(), await doc2.read()

    try:
        # Extract signatures
        if doc1.filename.lower().endswith(".pdf"):
            sig1 = extract_signature_from_pdf(b1)
        elif doc1.filename.lower().endswith(".docx"):
            sig1 = extract_signature_from_docx(b1)
        else:
            raise HTTPException(400, f"Unsupported type for doc1: {doc1.filename}")

        if doc2.filename.lower().endswith(".pdf"):
            sig2 = extract_signature_from_pdf(b2)
        elif doc2.filename.lower().endswith(".docx"):
            sig2 = extract_signature_from_docx(b2)
        else:
            raise HTTPException(400, f"Unsupported type for doc2: {doc2.filename}")

        # Clean signatures (morphological operations)
        cleaned_sig1 = clean_signature(sig1)
        cleaned_sig2 = clean_signature(sig2)
        
        # Calculate match accuracy with advanced method
        accuracy = compute_match_accuracy(cleaned_sig1, cleaned_sig2)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse({"match_accuracy": round(accuracy, 2)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)