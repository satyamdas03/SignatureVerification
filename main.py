## BETTER ALGO 3 (best one yet)
import io, os, math, tempfile, shutil
import numpy as np
import cv2
import fitz  # PyMuPDF
import docx2txt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Tuple, Optional, Dict
from skimage.morphology import skeletonize
from skimage.metrics import structural_similarity as ssim

app = FastAPI(version="4.3.0")

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
    wvals, vecs = np.linalg.eig(cov)
    major = vecs[:, np.argmax(wvals)]
    angle = math.degrees(math.atan2(major[1], major[0]))
    # rotate to near-horizontal within [-45,45]
    rot_angle = angle if abs(angle) <= 60 else (angle - 90 if angle > 0 else angle + 90)
    h, w_ = mask.shape
    M = cv2.getRotationMatrix2D((w_ // 2, h // 2), rot_angle, 1.0)
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
# Matching (robust & overlap-first, calibrated)
# -----------------------------
def _prep_for_match(mask: np.ndarray) -> np.ndarray:
    mask = _deskew_to_horizontal(mask)
    mask = _tight_crop(mask, pad=8)
    mask = _resize_pad(mask, (256, 256))
    return (mask > 0).astype(np.uint8) * 255


def _center_of_mass(mask: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments((mask > 0).astype(np.uint8))
    if m["m00"] == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return cx, cy


def _affine_scale(mask: np.ndarray, scale: float) -> np.ndarray:
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _affine_translate(mask: np.ndarray, tx: float, ty: float) -> np.ndarray:
    h, w = mask.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _rotate(mask: np.ndarray, angle: float) -> np.ndarray:
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _skeleton(mask: np.ndarray) -> np.ndarray:
    return skeletonize((mask > 0).astype(bool)).astype(np.uint8) * 255


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def _dice(a: np.ndarray, b: np.ndarray, r: int = 0) -> float:
    A = _dilate(a, r) > 0
    B = _dilate(b, r) > 0
    inter = np.logical_and(A, B).sum()
    s = A.sum() + B.sum()
    return float(2.0 * inter / s) if s > 0 else 0.0


def _iou(a: np.ndarray, b: np.ndarray, r: int = 0) -> float:
    A = _dilate(a, r) > 0
    B = _dilate(b, r) > 0
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return float(inter / union) if union > 0 else 0.0


def _largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _matchShapes_sim(a: np.ndarray, b: np.ndarray) -> float:
    ca = _largest_contour(a)
    cb = _largest_contour(b)
    if ca is None or cb is None:
        return 0.0
    m = cv2.matchShapes(ca, cb, cv2.CONTOURS_MATCH_I1, 0.0)  # 0 = identical
    return float(np.exp(-3.0 * m))  # distance → similarity


def _hu_similarity(a: np.ndarray, b: np.ndarray) -> float:
    m1 = cv2.moments((a > 0).astype(np.uint8))
    m2 = cv2.moments((b > 0).astype(np.uint8))
    if m1["m00"] == 0 or m2["m00"] == 0:
        return 0.0
    h1 = cv2.HuMoments(m1).flatten()
    h2 = cv2.HuMoments(m2).flatten()
    h1 = np.sign(h1) * np.log1p(np.abs(h1))
    h2 = np.sign(h2) * np.log1p(np.abs(h2))
    d = np.linalg.norm(h1 - h2)
    return float(np.exp(-0.75 * d))


def _chamfer_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ea = cv2.Canny(a, 40, 120)
    eb = cv2.Canny(b, 40, 120)
    dtA = cv2.distanceTransform(255 - ea, cv2.DIST_L2, 3)
    dtB = cv2.distanceTransform(255 - eb, cv2.DIST_L2, 3)
    a_on_b = dtB[ea > 0].mean() if np.any(ea > 0) else 1e3
    b_on_a = dtA[eb > 0].mean() if np.any(eb > 0) else 1e3
    d = (a_on_b + b_on_a) / 2.0
    diag = math.hypot(*a.shape)
    return float(max(0.0, 1.0 - (d / (0.06 * diag + 1e-6))))


def _blurred_ssim(a: np.ndarray, b: np.ndarray) -> float:
    # SSIM on blurred masks → tolerant to stroke thickness/background
    aa = cv2.GaussianBlur(a, (7, 7), 0).astype(np.float32) / 255.0
    bb = cv2.GaussianBlur(b, (7, 7), 0).astype(np.float32) / 255.0
    try:
        val = ssim(aa, bb, data_range=1.0)
    except Exception:
        val = 0.0
    return float(np.clip(val, 0.0, 1.0))


def _orb_affine_refine(ref: np.ndarray, mov: np.ndarray) -> Tuple[np.ndarray, float]:
    """Use ORB+RANSAC to estimate affine from mov→ref; return warped mov and inlier ratio."""
    aa = cv2.GaussianBlur(ref, (5, 5), 0)
    bb = cv2.GaussianBlur(mov, (5, 5), 0)
    orb = cv2.ORB_create(nfeatures=1000)
    k1, d1 = orb.detectAndCompute(aa, None)
    k2, d2 = orb.detectAndCompute(bb, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return mov, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d2, d1, k=2)  # mov→ref
    good = [(m.queryIdx, m.trainIdx) for m, n in matches if m.distance < 0.85 * n.distance]
    if len(good) < 6:
        return mov, 0.0
    src = np.float32([k2[i].pt for i, _ in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[j].pt for _, j in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0, maxIters=2000)
    if M is None or inliers is None:
        return mov, 0.0
    inlier_ratio = float(inliers.sum() / max(10, min(len(k1), len(k2))))
    warped = cv2.warpAffine(mov, M, (mov.shape[1], mov.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    return warped, float(np.clip(inlier_ratio, 0.0, 1.0))


def _coarse_align(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    # Scale sweep (area-based first, then refine) + rotation sweep; center each try
    a1 = (m1 > 0).sum()
    a2 = (m2 > 0).sum()
    area_scale = float(np.clip(np.sqrt(a1 / a2), 0.85, 1.20)) if (a1 > 0 and a2 > 0) else 1.0
    scales = np.linspace(area_scale * 0.95, area_scale * 1.05, 7)  # narrow around area_scale
    angles = list(range(-12, 13, 2))

    best = m2
    best_score = -1.0
    sk1 = _skeleton(m1)

    for s in scales:
        scaled = _affine_scale(m2, float(s))
        c1x, c1y = _center_of_mass(m1)
        c2x, c2y = _center_of_mass(scaled)
        translated = _affine_translate(scaled, c1x - c2x, c1y - c2y)
        for ang in angles:
            rot = _rotate(translated, float(ang))
            score = _dice(sk1, _skeleton(rot), r=2)
            if score > best_score:
                best_score = score
                best = rot
    return best


def _calibrate(base: float) -> float:
    """
    Perceptual calibration so 'pretty close' reads higher without lifting impostors.
    Monotonic: y = 1 - (1 - x)^gamma  with gamma≈1.8
    """
    return float(np.clip(1.0 - (1.0 - base) ** 1.8, 0.0, 1.0))


def compute_match_accuracy(sig1: np.ndarray, sig2: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Normalize & align
    m1 = _prep_for_match(sig1)
    m2 = _prep_for_match(sig2)
    m2 = _coarse_align(m1, m2)
    m2, orb_inlier = _orb_affine_refine(m1, m2)  # small geometric polish

    # Core similarities (more tolerant overlap)
    dice_mask = _dice(m1, m2, r=3)
    dice_skel = _dice(_skeleton(m1), _skeleton(m2), r=2)
    dice = max(dice_mask, dice_skel)

    iou = _iou(m1, m2, r=3)
    chamfer = _chamfer_similarity(m1, m2)
    shape_sim = _matchShapes_sim(m1, m2)
    hu = _hu_similarity(m1, m2)
    bssim = _blurred_ssim(m1, m2)

    # Weighted score (overlap-first; features supportive)
    base = (
        0.50 * dice +
        0.20 * bssim +
        0.15 * chamfer +
        0.08 * iou +
        0.05 * shape_sim +
        0.02 * hu
        # orb_inlier only used for alignment; if you want, add tiny weight:
        # + 0.00 * orb_inlier
    )

    # Keep impostors low: if all overlap cues are weak, squash score
    if (dice < 0.20) and (chamfer < 0.20) and (iou < 0.20):
        base *= 0.5

    calibrated = _calibrate(float(np.clip(base, 0.0, 1.0)))
    accuracy = float(np.clip(calibrated * 100.0, 0.0, 100.0))
    breakdown = {
        "dice": round(dice, 3),
        "bssim": round(bssim, 3),
        "chamfer": round(chamfer, 3),
        "iou": round(iou, 3),
        "shape": round(shape_sim, 3),
        "hu": round(hu, 3),
        "orb_inlier": round(orb_inlier, 3),
        "base_before_calibration": round(float(base), 3)
    }
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

