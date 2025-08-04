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







## pdf as input
import os, io, tempfile, shutil
import numpy as np
import cv2
import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import docx2txt
from skimage.metrics import structural_similarity as ssim

app = FastAPI(version="3.1.0")

# Fixed size for SSIM comparison
TARGET_SIZE = (300, 150)

# -------------------------------------------------------------------
# 1) Document → Signature Extraction
# -------------------------------------------------------------------

def extract_signature_from_pdf(pdf_bytes: bytes) -> np.ndarray:
    """
    Extract embedded images from the last PDF page, threshold each,
    and pick the one with the highest white-pixel ratio (the signature).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[-1]  # last page
    images = page.get_images(full=True)
    if not images:
        doc.close()
        raise ValueError("No images found in PDF")

    best_mask = None
    best_ratio = 0.0

    for img_meta in images:
        xref = img_meta[0]
        pix = fitz.Pixmap(doc, xref)
        # If CMYK, convert to RGB
        if pix.n > 4:
            pix = fitz.Pixmap(pix, 0)
        arr = np.frombuffer(pix.samples, np.uint8)
        channels = pix.n
        arr = arr.reshape((pix.height, pix.width, channels))
        pix = None

        # Convert to gray if needed
        if channels >= 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        # Threshold (invert: strokes→white)
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Compute stroke density
        ratio = cv2.countNonZero(mask) / (mask.size)
        if ratio > best_ratio:
            best_ratio = ratio
            best_mask = mask

    doc.close()
    if best_mask is None:
        raise ValueError("Could not extract signature from PDF images")

    return best_mask


def extract_signature_from_docx(docx_bytes: bytes) -> np.ndarray:
    """
    Extract all images from a DOCX, pick the largest by pixel area,
    and threshold to obtain a signature mask.
    """
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

    # pick largest by area
    _, _, best_img = max(candidates, key=lambda x: x[1])

    _, mask = cv2.threshold(
        best_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return mask

# -------------------------------------------------------------------
# 2) Preprocess & Resize
# -------------------------------------------------------------------

def preprocess_signature(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    resized = cv2.resize(clean, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized

# -------------------------------------------------------------------
# 3) Compute SSIM Match Accuracy
# -------------------------------------------------------------------

def compute_match_accuracy(sig1: np.ndarray, sig2: np.ndarray) -> float:
    score, _ = ssim(sig1, sig2, full=True)
    return float(round(score * 100, 2))

# -------------------------------------------------------------------
# 4) FastAPI Endpoint
# -------------------------------------------------------------------

@app.post("/verify-signature", response_class=JSONResponse)
async def verify_signature(
    doc1: UploadFile = File(..., description="First document (PDF or DOCX)"),
    doc2: UploadFile = File(..., description="Second document (PDF or DOCX)")
):
    b1, b2 = await doc1.read(), await doc2.read()

    try:
        # 4.1) Extract
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

        # 4.2) Preprocess & match
        m1 = preprocess_signature(sig1)
        m2 = preprocess_signature(sig2)
        accuracy = compute_match_accuracy(m1, m2)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse({"match_accuracy": accuracy})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




