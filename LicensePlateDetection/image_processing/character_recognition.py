"""Plate character segmentation and OCR.

Recognition uses glyph_matcher.TemplateMatcher: shape similarity across font
and labelled real glyphs. No unconditional per-character rewrite rules.
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .preprocessing import rgb_to_grayscale, unsharp_mask, enhance_contrast_dynamic
from .thresholding import otsu_threshold, adaptive_local_threshold
from .morphology import bridge_broken_characters
from .connected_components import find_connected_components


# Common layouts supported by this project; this is not a legal registry check.
VN_PLATE_REGEX = re.compile(r'^\d{2}[ABCDEFGHKLMNPSTUVXYZ]{1,2}\d?\d{3,6}$')


def is_valid_vn_plate(text: str) -> bool:
    if not text or text == 'KXD':
        return False
    parts = text.strip().split()
    if len(parts) >= 2:
        prefix = ''.join(parts[:-1]).replace('-', '')
        body = parts[-1].replace('.', '')
        return bool(re.fullmatch(r'\d{2}[ABCDEFGHKLMNPSTUVXYZ]{1,2}\d?', prefix)
                    and re.fullmatch(r'\d{3,6}', body))
    compact = re.sub(r'[- .]', '', text)
    return bool(VN_PLATE_REGEX.fullmatch(compact))


def format_plate_display(text: str, plate_type: Optional[str] = None) -> str:
    """Định dạng chuỗi biển số hiển thị theo yêu cầu:
    Thay toàn bộ dấu gạch ngang (-) thành dấu cách (khoảng trống) để các cụm số và chữ dễ nhìn.
    - Biển 1 dòng: ví dụ '30F 123.45', '29A 146.08', '51G 1234'
    - Biển 2 dòng: ví dụ '18 MD2 074.73', '29 A1 123.45', '59 G2 622.33'
    """
    if not text or text == "KXD":
        return text or "KXD"

    if '-' in text:
        text = text.replace('-', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if not clean:
        return text

    if plate_type == 'double_row':
        m = re.match(r'^(\d{2})([A-Z]{1,2}\d?)(.*)$', clean)
        if m:
            prov, ser, num = m.group(1), m.group(2), m.group(3)
            if len(num) == 5:
                num = f"{num[:3]}.{num[3:]}"
            return f"{prov} {ser} {num}".strip()

    m1 = re.match(r'^(\d{2}[A-Z]{1,2})(\d{3,6})$', clean)
    if m1:
        prefix, num = m1.group(1), m1.group(2)
        if len(num) == 5:
            num = f"{num[:3]}.{num[3:]}"
        return f"{prefix} {num}"

    return text.replace('-', ' ')


def tight_crop_binary(arr: np.ndarray) -> np.ndarray:
    """Cắt sát viền pixel trắng (bounding box chặt nhất)."""
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    if not np.any(rows) or not np.any(cols):
        return arr
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return arr[ymin:ymax+1, xmin:xmax+1]


def normalize_char_patch(arr: np.ndarray, target_size=(20, 32)) -> np.ndarray:
    """Chuẩn hóa patch ký tự nhị phân về kích thước cố định W=20, H=32 sau khi tight crop."""
    tight = tight_crop_binary(arr)
    if tight.shape[0] < 2 or tight.shape[1] < 2:
        return np.zeros((target_size[1], target_size[0]), dtype=bool)

    im = Image.fromarray((tight.astype(np.uint8) * 255), mode="L")
    resized = im.resize(target_size, Image.Resampling.BILINEAR)
    return (np.array(resized) > 100)


def D(P: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> float:
    """
    Mật độ diện tích cục bộ (Area Density Metric - D):
    D(y1, y2, x1, x2) = 1 / ((y2 - y1 + 1)(x2 - x1 + 1)) * sum(P[y, x])
    với 0 <= y1 <= y2 <= 31 và 0 <= x1 <= x2 <= 19.
    """
    sub = P[y1:y2+1, x1:x2+1]
    return float(np.mean(sub)) if sub.size > 0 else 0.0


def calculate_metrics(p_norm: np.ndarray) -> Dict[str, Any]:
    """
    Trích xuất các vectơ chiếu và độ đối xứng toàn cục:
    1. Horizontal Projection (HP)
    2. Vertical Projection (VP)
    3. Horizontal Symmetry (Sym_X)
    4. Vertical Symmetry (Sym_Y)
    """
    P = p_norm.astype(float)
    W, H = 20, 32
    HP = P.sum(axis=1)
    VP = P.sum(axis=0)

    # Đối xứng ngang Sym_X (qua trục x = 9.5)
    diff_x = np.abs(P[:, :10] - P[:, 19:9:-1])
    sym_x = float(np.sum(diff_x)) / (W * H)

    # Đối xứng dọc Sym_Y (qua trục y = 15.5)
    diff_y = np.abs(P[:16, :] - P[31:15:-1, :])
    sym_y = float(np.sum(diff_y)) / (W * H)

    return {
        "P": P,
        "HP": HP,
        "VP": VP,
        "sym_x": sym_x,
        "sym_y": sym_y,
    }


def compute_euler_holes(patch_20x32: np.ndarray) -> int:
    """
    Đếm lỗ rỗng khép kín quan sát được; không suy lỗ từ vùng tâm rỗng
    (Euler Noise Resilience theo Mục 2 của tài liệu nghiệp vụ):
    - 2 lỗ: 8, B
    - 1 lỗ: 0, 4, 6, 9, A, D, P, Q
    - 0 lỗ: 1, 2, 3, 5, 7, C, E, F, G, H, K, L, M, N, S, T, U, V, X, Y, Z
    """
    try:
        arr = (patch_20x32.astype(np.uint8) * 255)
        padded = np.pad(arr, 1, mode='constant', constant_values=0)
        contours, hierarchy = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            raw_holes = 0
        else:
            holes = 0
            for h_info in hierarchy[0]:
                if h_info[3] != -1:
                    holes += 1
            raw_holes = min(2, holes)

        return raw_holes
    except Exception:
        return 0


from .glyph_matcher import TemplateMatcher


_GLOBAL_MATCHER = None

def get_template_matcher() -> TemplateMatcher:
    global _GLOBAL_MATCHER
    if _GLOBAL_MATCHER is None:
        _GLOBAL_MATCHER = TemplateMatcher()
    return _GLOBAL_MATCHER


def binarize_cropped_plate(cropped_pil: Image.Image) -> np.ndarray:
    """Nhị phân hóa thích nghi và bảo toàn nét chữ trên lòng biển số xe."""
    arr = np.array(cropped_pil)
    if arr.ndim == 3 and arr.shape[2] == 3:
        r, g, b = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), arr[:, :, 2].astype(float)
        is_yellow = (np.mean(r) > 80 and np.mean(g) > 70 and (np.mean(r) + np.mean(g)) / 2.0 > np.mean(b) * 1.15)
        if is_yellow:
            gray = np.clip((r * 0.45 + g * 0.55), 0, 255).astype(np.uint8)
        else:
            gray = rgb_to_grayscale(arr)
    else:
        gray = rgb_to_grayscale(arr)
    unsharp = unsharp_mask(gray, amount=1.5, sigma=1.0)
    # On a nearly flat background, percentile clipping can erase a sparse glyph
    # while amplifying its sharpening halo. Preserve the full range in that case.
    raw_lo, raw_hi = np.percentile(gray, (2, 98))
    if raw_hi - raw_lo < .25 * float(np.ptp(gray)) and float(np.ptp(gray)) > 40:
        lo, hi = float(unsharp.min()), float(unsharp.max())
        enhanced = np.clip((unsharp.astype(np.float32)-lo)*255.0/max(hi-lo, 1.0), 0, 255).astype(np.uint8)
    else:
        enhanced = enhance_contrast_dynamic(unsharp)

    h, w = enhanced.shape
    pad_y = max(2, int(round(h * 0.06)))
    pad_x = max(2, int(round(w * 0.05)))
    inner = enhanced[pad_y : h - pad_y, pad_x : w - pad_x]

    if inner.size > 0:
        _, t_otsu = otsu_threshold(inner)
    else:
        _, t_otsu = otsu_threshold(enhanced)

    p25 = float(np.percentile(inner, 25)) if inner.size > 0 else float(np.percentile(enhanced, 25))
    mean_inner = float(np.mean(inner)) if inner.size > 0 else float(np.mean(enhanced))

    # Phát hiện nền tối / nền sáng để chọn chiến lược ngưỡng (Lỗi 3)
    if mean_inner > 160:  # Nền sáng (biển trắng/vàng bình thường)
        t_opt = int(round(0.80 * t_otsu + 0.20 * p25))
        t_opt = max(40, min(200, t_opt))
    else:  # Nền tối / thiếu sáng
        t_opt = int(round(0.65 * t_otsu + 0.35 * p25))
        t_opt = max(25, min(140, t_opt))

    char_bin = (enhanced < t_opt)

    # Khử viền mép ngoài biển số
    b_y = max(2, int(round(h * 0.04)))
    b_x = max(2, int(round(w * 0.03)))
    char_bin[:b_y, :] = False
    char_bin[-b_y:, :] = False
    char_bin[:, :b_x] = False
    char_bin[:, -b_x:] = False

    return char_bin


def estimate_tilt_angle(candidates: List[Dict[str, Any]], h: int) -> float:
    """Ước lượng góc nghiêng của biển số xe bằng độ dốc trục hoành của các ký tự."""
    if len(candidates) < 3:
        return 0.0

    mid_y = h * 0.50
    top_row = [c for c in candidates if c["cy"] < mid_y]
    bot_row = [c for c in candidates if c["cy"] >= mid_y]

    target_row = top_row if len(top_row) >= 3 else (bot_row if len(bot_row) >= 3 else candidates)
    if len(target_row) < 3:
        return 0.0

    target_row.sort(key=lambda c: c["cx"])
    dx = target_row[-1]["cx"] - target_row[0]["cx"]
    dy = target_row[-1]["cy"] - target_row[0]["cy"]
    if abs(dx) < 1e-3:
        return 0.0

    angle = np.degrees(np.arctan2(dy, dx))
    return float(angle)


def segment_characters(char_bin: np.ndarray, plate_type: str = "double_row") -> List[Dict[str, Any]]:
    """Phân đoạn các ký tự bên trong lòng biển số."""
    h, w = char_bin.shape
    components, character_labels = find_connected_components(char_bin, connectivity=8, compute_quad=False)

    candidates = []
    min_h = int(round(h * 0.16))
    max_h = int(round(h * 0.88)) if plate_type == "single_row" else int(round(h * 0.65))

    for c in components:
        bx, by, bw, bh = c["bbox"]
        area = c["bbox_area"]
        density = c["density"]

        if bh < min_h or bh > max_h:
            continue
        if bw < 3 or bw > int(round(w * 0.40)):
            continue
        # A narrow line touching the cleared crop edge is a frame fragment,
        # not a new first/last character. Keep ordinary thin 1 inside the plate.
        edge_x=max(2,round(w*.025))
        if (bx<=edge_x or bx+bw>=w-edge_x) and bw/max(1,bh)<.18:
            continue
        if area < 25 or area > (h * w * 0.35):
            continue

        ar = float(bw) / float(bh)
        if not (0.07 <= ar <= 2.20):  # Hạ từ 0.10 -> 0.07 để giữ ký tự '1' rất mảnh (Lỗi 2)
            continue
        if not (0.15 <= density <= 0.95):
            continue

        patch = (character_labels[by : by + bh, bx : bx + bw] == c["id"])
        candidates.append({
            "bbox": (bx, by, bw, bh),
            "cx": bx + bw / 2.0,
            "cy": by + bh / 2.0,
            "area": area,
            "patch": patch
        })

    # Khử đè lặp (NMS)
    kept = []
    candidates.sort(key=lambda c: c["area"], reverse=True)
    for c in candidates:
        bx, by, bw, bh = c["bbox"]
        dup = False
        for k in kept:
            kx, ky, kw, kh = k["bbox"]
            xi1 = max(bx, kx)
            yi1 = max(by, ky)
            xi2 = min(bx + bw, kx + kw)
            yi2 = min(by + bh, ky + kh)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            iou = inter / float(bw * bh + kw * kh - inter) if (bw * bh + kw * kh - inter) > 0 else 0
            if iou > 0.45:
                dup = True
                break
        if not dup:
            kept.append(c)

    return kept


def _repair_row(row, mask, matcher, digit_only, max_characters=5):
    if not row:
        return row
    median_y=float(np.median([c['bbox'][1] for c in row]))
    median_h=float(np.median([c['bbox'][3] for c in row]))
    median_cy=float(np.median([c['cy'] for c in row]))
    result=[]
    ordered=sorted(row,key=lambda c:c['cx'])
    median_width=float(np.median([c['bbox'][2] for c in row]))
    for item in ordered:
        c=dict(item);x,y,w,h=c['bbox']
        # Border pieces may be thick, diagonal or embossed. Test their position
        # and separation from the text, rather than assuming all are thin lines.
        at_left=x<mask.shape[1]*.065
        at_right=x+w>mask.shape[1]*.935
        isolated=False
        if len(ordered)>=4 and item is ordered[0]:
            isolated=ordered[1]['bbox'][0]-(x+w)>1.25*median_width
        elif len(ordered)>=4 and item is ordered[-1]:
            prev=ordered[-2]['bbox'];isolated=x-(prev[0]+prev[2])>1.25*median_width
        if (at_left or at_right) and (isolated or (h<.72*median_h and abs(c['cy']-median_cy)>.20*median_h)):
            continue
        # A skinny frame fragment outside the text baseline is not digit 1.
        if w/h<.17 and (x<mask.shape[1]*.20 or x+w>mask.shape[1]*.80) and abs(c['cy']-median_cy)>.25*median_h:
            continue
        if len(row)>=3 and h>1.25*median_h and y<median_y-.20*median_h:
            y=max(0,round(median_y));h=min(mask.shape[0]-y,max(2,round(median_h)))
            c['bbox']=(x,y,w,h);c['patch']=mask[y:y+h,x:x+w]
            c['cx']=x+w/2;c['cy']=y+h/2
        patch=c['patch'];ph,pw=patch.shape
        if pw/ph>.70 and pw>1.35*median_width and len(row)<max_characters:
            projection=patch.sum(axis=0)
            left=max(2,round(pw*.25));right=min(pw-2,round(pw*.75))
            if right>left:
                cut=left+int(np.argmin(projection[left:right]))
                if projection[cut]<.35*max(1,projection.max()):
                    pieces=[patch[:,:cut],patch[:,cut:]]
                    if all(piece.sum()>ph*2 and piece.shape[1]/ph>.12 for piece in pieces):
                        whole=matcher.match_details(patch,is_digit_only=digit_only)['score']
                        split=np.mean([matcher.match_details(piece,is_digit_only=digit_only)['score'] for piece in pieces])
                        if split>whole+.025:
                            for offset,piece in [(0,pieces[0]),(cut,pieces[1])]:
                                q=dict(c);q['patch']=piece;q['bbox']=(int(x+offset),int(y),int(piece.shape[1]),int(ph))
                                q['cx']=x+offset+piece.shape[1]/2;q['cy']=y+ph/2
                                result.append(q)
                            continue
        result.append(c)
    return sorted(result,key=lambda c:c['cx'])


def _match_context(matcher, patch, is_digit_only=False, is_letter_only=False):
    """Use syntax only as weak evidence, never force a clearly different glyph."""
    return _apply_context(matcher.match_details(patch),is_digit_only,is_letter_only)


def _apply_context(result,is_digit_only=False,is_letter_only=False):
    scores=result.get('class_scores', {})
    if not scores or not (is_digit_only or is_letter_only):
        return result
    allowed={ch:score for ch,score in scores.items()
             if ch.isdigit()==is_digit_only}
    best=max(allowed,key=allowed.get)
    raw=result['char']
    conflict=raw not in allowed
    result['context_applied']=False
    result['syntax_conflict']=conflict
    if conflict and result['score']-allowed[best]<=.03:
        result['char']=best
        result['score']=allowed[best]
        result['context_applied']=True
        result['needs_review']=True
        result['proposed_char']=best
        result['syntax_conflict']=False
    return result


def _strip_plate_frame(mask):
    """Remove long border lines, not short strokes inside individual glyphs."""
    h,w=mask.shape
    image=mask.astype(np.uint8)
    vertical=cv2.morphologyEx(image,cv2.MORPH_OPEN,np.ones((max(3,round(h*.70)),1),np.uint8))
    horizontal=cv2.morphologyEx(image,cv2.MORPH_OPEN,np.ones((1,max(3,round(w*.65))),np.uint8))
    outer_v=np.zeros_like(image);outer_v[:,:max(2,round(w*.18))]=1;outer_v[:,-max(2,round(w*.18)):]=1
    outer_h=np.zeros_like(image);outer_h[:max(2,round(h*.18))]=1;outer_h[-max(2,round(h*.18)):]=1
    frame=(vertical & outer_v)|(horizontal & outer_h)
    frame=cv2.dilate(frame,np.ones((3,3),np.uint8))>0
    return mask & ~frame


def _mask_candidates(image):
    """Independent foreground masks; no union of all methods and noise."""
    gray=np.array(image.convert('L'))
    masks=[('enhanced_otsu',binarize_cropped_plate(image))]
    _,otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    masks.append(('plain_otsu',otsu>0))
    block=max(15,int(round(gray.shape[0]*.25))|1)
    adaptive=adaptive_local_threshold(gray,block_size=block,k=.18,method='sauvola')==0
    masks.append(('sauvola',adaptive))
    # One-pixel repair at the standard crop scale, not a plate-wide closing.
    masks.append(('sauvola_repair',cv2.morphologyEx(adaptive.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))>0))
    # Embossed chrome plates need local relief rather than a global dark class.
    local=cv2.createCLAHE(clipLimit=2.,tileGridSize=(4,4)).apply(gray)
    emboss=cv2.adaptiveThreshold(local,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,5)
    masks.append(('local_relief',emboss>0))
    augmented=[]
    for name,mask in masks:
        if name in ('enhanced_otsu','plain_otsu'):
            augmented.append((name+'_frame_clean',_strip_plate_frame(mask)))
        augmented.append((name,mask))
    result=[];seen=set()
    for name,mask in augmented:
        mask=mask.copy()
        by,bx=max(2,round(mask.shape[0]*.035)),max(2,round(mask.shape[1]*.025))
        mask[:by]=False;mask[-by:]=False;mask[:,:bx]=False;mask[:,-bx:]=False
        key=np.packbits(mask).tobytes()
        if key not in seen:
            result.append((name,mask));seen.add(key)
    return result


def _recognition_quality(result, plate_type):
    chars=result.get('characters',[])
    if not chars:
        return -1.
    if plate_type=='double_row':
        nt=sum(c['row']==1 for c in chars);nb=sum(c['row']==2 for c in chars)
        complete=3<=nt<=5 and 4<=nb<=5
    else:
        complete=7<=len(chars)<=9
    confidence=result.get('avg_confidence',0.)
    conflicts=sum(c.get('match_details',{}).get('syntax_conflict',False) for c in chars)/len(chars)
    uncertain=sum(c.get('match_details',{}).get('needs_review',True) for c in chars)/len(chars)
    # Distinct glyph components should not occupy the same bounding area.
    # A connected frame can surround a real glyph, creating a high-scoring
    # phantom letter and misleading mask fusion if chosen as the anchor.
    collision=0.
    for index,a in enumerate(chars):
        ax,ay,aw,ah=a['bbox']
        for b in chars[index+1:]:
            if a['row']!=b['row']:
                continue
            bx,by,bw,bh=b['bbox']
            overlap=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
            collision+=overlap/max(1,min(aw*ah,bw*bh))
    return (confidence + .18*complete - .08*uncertain - .08*conflicts
            + .02*is_valid_vn_plate(result.get('plate_text',''))
            + .06*result.get('ink_coverage',0.) - .20*collision)


def _format_characters(characters,plate_type):
    if plate_type=='double_row':
        top=''.join(c['char'] for c in characters if c['row']==1)
        bottom=''.join(c['char'] for c in characters if c['row']==2)
        prefix=top[:2]+' '+top[2:] if top[:2].isdigit() and len(top)>2 else top
        body=bottom[:3]+'.'+bottom[3:] if len(bottom)==5 else bottom
        return (prefix+' '+body).strip() or 'KXD'
    raw=''.join(c['char'] for c in characters)
    match=re.fullmatch(r'(\d{2}[A-Z]{1,2})(\d{4,5})',raw)
    if not match:
        return raw or 'KXD'
    prefix,body=match.groups()
    return prefix+' '+(body[:3]+'.'+body[3:] if len(body)==5 else body)


def _recognize_single_orientation(cropped_pil, plate_type='double_row', geometry_rectified=False):
    if cropped_pil.height<100:
        scale=120/cropped_pil.height
        cropped_pil=cropped_pil.resize((max(1,round(cropped_pil.width*scale)),120),Image.Resampling.BICUBIC)
    results=[]
    for name,mask in _mask_candidates(cropped_pil):
        result=_recognize_from_mask(cropped_pil,plate_type,geometry_rectified,mask_override=mask)
        result['binarization']=name
        results.append(result)
    # Prefer a segmentation supported by several methods, rather than choosing
    # a slightly higher average after accidentally dropping a difficult glyph.
    def row_counts(result):
        chars=result.get('characters',[])
        return tuple(sum(c['row']==row for c in chars) for row in ([1,2] if plate_type=='double_row' else [1]))
    from collections import Counter
    complete=[r for r in results if ((3<=row_counts(r)[0]<=5 and 4<=row_counts(r)[1]<=5) if plate_type=='double_row' else 7<=row_counts(r)[0]<=9)]
    pool=complete or results
    counts=Counter(row_counts(r) for r in pool)
    modal,support=counts.most_common(1)[0]
    consistent=[r for r in pool if row_counts(r)==modal] if support>=2 else pool
    best=max(consistent,key=lambda r:_recognition_quality(r,plate_type))
    if len(consistent)<2 or not best.get('characters'):
        return best
    fused=[]
    for idx,anchor in enumerate(best['characters']):
        ax,ay,aw,ah=anchor['bbox']
        votes=[]
        for alternative in consistent:
            options=[]
            for char in alternative.get('characters',[]):
                if char['row']!=anchor['row']:continue
                x,y,w,h=char['bbox']
                overlap=max(0,min(ax+aw,x+w)-max(ax,x))*max(0,min(ay+ah,y+h)-max(ay,y))
                iou=overlap/max(1,aw*ah+w*h-overlap)
                if iou>.4:options.append((iou,char))
            if options:votes.append(max(options,key=lambda pair:pair[0])[1]['match_details'])
        all_classes=set.intersection(*(set(v.get('class_scores',{})) for v in votes)) if votes else set()
        if len(votes)<2 or not all_classes:
            fused.append(anchor);continue
        scores={ch:float(np.mean([v['class_scores'][ch] for v in votes])) for ch in all_classes}
        ranked=sorted(scores.items(),key=lambda pair:(-pair[1],pair[0]));char,score=ranked[0]
        margin=score-ranked[1][1]
        details={'char':char,'score':round(score,4),'raw_char':char,'proposed_char':char,
                 'class_scores':scores,'rule_applied':False,'backend':'shape-v2',
                 'margin':round(margin,4),
                   'needs_review':score<.65 or margin<.025 or len({v['char'] for v in votes})>1,
                   'mask_disagreement':len({v['char'] for v in votes})>1,
                 'top_candidates':[{'char':ch,'score':round(sc,4)} for ch,sc in ranked[:3]]}
        row_chars=[c for c in best['characters'] if c['row']==anchor['row']]
        position=row_chars.index(anchor)
        if plate_type=='double_row' and anchor['row']==2:
            digit,letter=True,False
        elif plate_type=='single_row':
            digit,letter=position!=2,position==2
        else:
            digit=position in ([0,1,4] if len(row_chars)>=5 else [0,1])
            letter=position in ([2,3] if len(row_chars)>=5 else [2])
        details=_apply_context(details,digit,letter)
        copy=dict(anchor,char=details['char'],score=details['score'],match_details=details)
        fused.append(copy)
    best=dict(best)
    best['characters']=fused
    best['plate_text']=_format_characters(fused,plate_type)
    best['avg_confidence']=round(float(np.mean([c['score'] for c in fused])),3)
    best['uncertain_positions']=[i for i,c in enumerate(fused) if c['match_details'].get('needs_review',True)]
    best['needs_review']=bool(best['uncertain_positions']) or not is_valid_vn_plate(best['plate_text'])
    best['binarization_methods']=[r['binarization'] for r in consistent]
    return best


def _recognize_from_mask(
    cropped_pil: Image.Image,
    plate_type: str = "double_row",
    geometry_rectified: bool = False,
    mask_override=None
) -> Dict[str, Any]:
    """Quy trình nhận diện ký tự trên một hướng ảnh cụ thể."""
    # [P07]: Chuẩn hóa kích thước crop nhỏ (upscale giữ tỷ lệ nếu H < 100)
    if cropped_pil.height < 100:
        scale = 120.0 / float(cropped_pil.height)
        new_w = max(1, int(round(cropped_pil.width * scale)))
        new_h = 120
        cropped_pil = cropped_pil.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    char_bin = binarize_cropped_plate(cropped_pil) if mask_override is None else mask_override.copy()
    h, w = char_bin.shape
    candidates = segment_characters(char_bin, plate_type=plate_type)

    # [P08]: Bổ sung nhị phân hóa cục bộ Sauvola khi Otsu toàn cục bắt thiếu ký tự
    if len(candidates) < 4 and mask_override is None:
        gray_crop = np.array(cropped_pil.convert("L"))
        bs = max(15, int(round(cropped_pil.height * 0.35)))
        if bs % 2 == 0:
            bs += 1
        # The helper returns a binary image, not the local threshold matrix.
        sauv_bin = adaptive_local_threshold(gray_crop, block_size=bs, k=0.18, method="sauvola") == 0
        by, bx = max(2, round(sauv_bin.shape[0] * 0.04)), max(2, round(sauv_bin.shape[1] * 0.03))
        sauv_bin[:by, :] = sauv_bin[-by:, :] = False
        sauv_bin[:, :bx] = sauv_bin[:, -bx:] = False
        cands_sauv = segment_characters(sauv_bin, plate_type=plate_type)
        if len(cands_sauv) > len(candidates):
            candidates = cands_sauv
            char_bin = sauv_bin
            h, w = char_bin.shape

    tilt_cands = estimate_tilt_angle(candidates, h)
    needs_deskew = (abs(tilt_cands) >= 2.5) or (len(candidates) < 6)

    if needs_deskew and not geometry_rectified and mask_override is None:
        test_angles = [0.0]
        for a in [-tilt_cands, tilt_cands, -18.0, -15.0, -12.0, 12.0, 15.0, 18.0]:
            if abs(a) >= 2.5 and not any(abs(a - ta) < 1.0 for ta in test_angles):
                test_angles.append(a)

        best_score = -1.0
        best_cands = candidates
        best_bin = char_bin
        best_pil = cropped_pil

        # [P04]: Ước lượng màu nền sáng để đệm viền xoay, chống cắt mép và chống sinh rác đen
        arr_c = np.array(cropped_pil)
        if arr_c.ndim == 3:
            bg_val = tuple(np.percentile(arr_c, 75, axis=(0, 1)).astype(int))
        else:
            bg_val = int(np.percentile(arr_c, 75))

        for ang in test_angles:
            rot_pil = cropped_pil.rotate(ang, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=bg_val) if abs(ang) > 0.5 else cropped_pil
            rot_bin = binarize_cropped_plate(rot_pil) if abs(ang) > 0.5 else char_bin
            cands_rot = segment_characters(rot_bin, plate_type=plate_type)

            if plate_type == "double_row" and len(cands_rot) >= 4:
                all_cys = [c['cy'] for c in cands_rot]
                s_y = (min(all_cys) + float(np.median(all_cys))) / 2.0
                top_c = [c for c in cands_rot if c['cy'] < s_y]
                bot_c = [c for c in cands_rot if c['cy'] >= s_y]
                sc = 0.0
                if 3 <= len(top_c) <= 5: sc += 3.5
                if 4 <= len(bot_c) <= 6: sc += 4.5
                vt = np.var([c['cy'] for c in top_c]) if len(top_c) >= 3 else 999.0
                vb = np.var([c['cy'] for c in bot_c]) if len(bot_c) >= 3 else 999.0
                if len(top_c) >= 3: sc += max(0.0, 2.0 - vt / 40.0)
                if len(bot_c) >= 3: sc += max(0.0, 2.0 - vb / 40.0)
            else:
                sc = float(len(cands_rot))
                if len(cands_rot) >= 5:
                    sc += max(0.0, 2.0 - np.var([c['cy'] for c in cands_rot]) / 40.0)

            if sc > best_score:
                best_score = sc
                best_cands = cands_rot
                best_bin = rot_bin
                best_pil = rot_pil

        if best_score >= 5.0 or len(best_cands) > len(candidates):
            candidates = best_cands
            char_bin = best_bin
            cropped_pil = best_pil
            h, w = char_bin.shape

    # [P05]: Xác nhận lại bố cục một/hai dòng dựa trên phân bố ký tự sau khi nắn thẳng
    if len(candidates) >= 4:
        cys = [c["cy"] for c in candidates]
        y_span = max(cys) - min(cys)
        c_h = np.median([c["bbox"][3] for c in candidates])
        if plate_type == "double_row" and y_span < 0.28 * c_h and (w / float(h)) >= 2.2:
            plate_type = "single_row"
        elif plate_type == "single_row" and y_span > 0.85 * c_h and (w / float(h)) < 2.5:
            plate_type = "double_row"

    matcher = get_template_matcher()

    if not candidates or len(candidates) < 2:
        return {
            "plate_text": "KXD",
            "needs_review": True,
            "uncertain_positions": [],
            "recognizer": "shape-v2",
            "char_count": len(candidates),
            "avg_confidence": 0.0,
            "characters": [],
            "deskewed_image": cropped_pil
        }

    recognized_chars = []

    if plate_type == "double_row":
        # Phân tách 2 hàng bằng khe hở lớn nhất (Max Vertical Gap) giữa 2 cụm ký tự thay vì (min+max)/2
        all_cys = sorted([c["cy"] for c in candidates])
        if len(all_cys) >= 4:
            gaps = [(all_cys[i+1] - all_cys[i], (all_cys[i+1] + all_cys[i]) / 2.0) for i in range(len(all_cys) - 1)]
            y_span = max(all_cys) - min(all_cys)
            valid_gaps = [g for g in gaps if (min(all_cys) + 0.20 * y_span) <= g[1] <= (min(all_cys) + 0.80 * y_span)]
            if valid_gaps:
                split_y = max(valid_gaps, key=lambda g: g[0])[1]
            else:
                split_y = (min(all_cys) + max(all_cys)) / 2.0
        else:
            split_y = (min(all_cys) + max(all_cys)) / 2.0
        top_row = [c for c in candidates if c["cy"] < split_y]
        bot_row = [c for c in candidates if c["cy"] >= split_y]

        top_row.sort(key=lambda c: c["cx"])
        bot_row.sort(key=lambda c: c["cx"])

        # Lọc đốm nhiễu ốc vít / bóng viền nếu số ký tự quá nhiều
        if top_row:
            min_top_x = min([c['cx'] for c in top_row])
        else:
            min_top_x = 0

        if len(top_row) > 5:
            med_h = np.median([c['bbox'][3] for c in top_row])
            top_row = [c for c in top_row if med_h * 0.70 <= c['bbox'][3] <= med_h * 1.35]
            if len(top_row) > 5:
                top_row = top_row[:5]

        if len(bot_row) > 5:
            med_h = np.median([c['bbox'][3] for c in bot_row])
            bot_row = [
                c for c in bot_row
                if (med_h * 0.70 <= c['bbox'][3] <= med_h * 1.35) and (c['cx'] >= min_top_x - 30)
            ]
            if len(bot_row) > 5:
                bot_row = bot_row[:5]

        # Repair rows without discarding the left side of a touching pair.
        # The old artifact rule could turn a connected "60" into just "0".
        top_row = _repair_row(top_row, char_bin, matcher, digit_only=False)
        bot_row = _repair_row(bot_row, char_bin, matcher, digit_only=True)

        top_text = ""
        top_scores = []
        for idx, c in enumerate(top_row):
            # Cú pháp dòng 1:
            # - Vị trí 0, 1: Bắt buộc là số (Mã tỉnh) -> is_digit_only=True
            # - Vị trí 2: Bắt buộc là chữ cái (Series) -> is_letter_only=True
            # - Vị trí 3:
            #   + Nếu có 5 ký tự (18-MD2): vị trí 3 là chữ, vị trí 4 là số
            #   + Nếu có 4 ký tự: vị trí 3 có thể là chữ (98-AA, 37-AC) hoặc số (34-B4, 90-B3)
            if len(top_row) >= 5:
                is_digit = (idx in [0, 1, 4])
                is_letter = (idx in [2, 3])
            else:
                is_digit = (idx in [0, 1])
                is_letter = (idx == 2)

            match_info = _match_context(matcher,c["patch"],is_digit_only=is_digit,is_letter_only=is_letter)
            ch, score = match_info["char"], match_info["score"]
            top_text += ch
            top_scores.append(score)
            recognized_chars.append({"char": ch, "score": score, "row": 1, "bbox": c["bbox"], "match_details": match_info})

        bot_text = ""
        bot_scores = []
        for idx, c in enumerate(bot_row):
            # Dòng 2 bắt buộc 100% là chữ số
            match_info = _match_context(matcher,c["patch"],is_digit_only=True)
            ch, score = match_info["char"], match_info["score"]
            bot_text += ch
            bot_scores.append(score)
            recognized_chars.append({"char": ch, "score": score, "row": 2, "bbox": c["bbox"], "match_details": match_info})

        if len(top_text) >= 3 and len(bot_text) >= 4:
            if len(bot_text) == 5:
                formatted_bot = f"{bot_text[:3]}.{bot_text[3:]}"
            else:
                formatted_bot = bot_text

            if len(top_text) >= 4 and top_text[2:4].isalpha():
                formatted_top = f"{top_text[:2]}-{top_text[2:]}"
            elif len(top_text) >= 4:
                formatted_top = f"{top_text[:2]}-{top_text[2:]}"
            else:
                formatted_top = f"{top_text[:2]}-{top_text[2:]}"

            if not top_text[:2].isdigit():
                formatted_top = top_text
            plate_text = f"{formatted_top} {formatted_bot}"
        else:
            plate_text = f"{top_text} {bot_text}" if (top_text or bot_text) else "KXD"

        all_scores = top_scores + bot_scores
        avg_score = float(np.mean(all_scores)) if all_scores else 0.0

    else:  # single_row (Biển 1 dòng)
        if len(candidates) >= 4:
            med_cy = float(np.median([c['cy'] for c in candidates]))
            med_h = float(np.median([c['bbox'][3] for c in candidates]))
            candidates = [
                c for c in candidates
                if abs(c['cy'] - med_cy) <= 0.35 * med_h and abs(c['bbox'][3] - med_h) <= 0.40 * med_h
            ]

        candidates = _repair_row(candidates, char_bin, matcher, digit_only=False, max_characters=9)
        candidates.sort(key=lambda c: c["cx"])
        plate_text_chars = []
        scores = []
        for idx, c in enumerate(candidates):
            is_digit = (idx < 2 or idx > 2)
            is_letter = (idx == 2)

            match_info = _match_context(matcher,c["patch"],is_digit_only=is_digit,is_letter_only=is_letter)
            ch, score = match_info["char"], match_info["score"]
            plate_text_chars.append(ch)
            scores.append(score)
            recognized_chars.append({"char": ch, "score": score, "row": 1, "bbox": c["bbox"], "match_details": match_info})

        raw_text = "".join(plate_text_chars)
        if len(raw_text) >= 6:
            prefix = raw_text[:3]
            body = raw_text[3:]
            if len(body) >= 5:
                plate_text = f"{prefix}-{body[:3]}.{body[3:]}"
            else:
                plate_text = f"{prefix}-{body}"
        else:
            plate_text = raw_text if raw_text else "KXD"

        avg_score = float(np.mean(scores)) if scores else 0.0

    covered=np.zeros_like(char_bin,dtype=bool)
    for character in recognized_chars:
        bx,by,bw,bh=character['bbox']
        covered[by:by+bh,bx:bx+bw]=True
    coverage=float(np.sum(char_bin & covered)/max(1,np.sum(char_bin)))
    uncertain = [i for i,c in enumerate(recognized_chars)
                 if c.get("match_details", {}).get("needs_review", True)]
    return {
        "ink_coverage": coverage,
        "needs_review": bool(uncertain) or not is_valid_vn_plate(plate_text),
        "uncertain_positions": uncertain,
        "recognizer": "shape-v2",
        "plate_text": plate_text,
        "char_count": len(recognized_chars),
        "avg_confidence": round(avg_score, 3),
        "characters": recognized_chars,
        "deskewed_image": cropped_pil
    }


def recognize_license_plate_template(cropped_pil, plate_type: str = "double_row", geometry_rectified: bool = False) -> Dict[str, Any]:
    if isinstance(cropped_pil, np.ndarray):
        cropped_pil = Image.fromarray(cropped_pil)
    """
    Hàm nhận diện toàn diện biển số xe (kèm cơ chế tự động chống lộn ngược ảnh 180°):
    - Đánh giá nhận dạng ở hướng 0 độ và 180 độ.
    - Ưu tiên tuyệt đối hướng khớp Regex biển số chuẩn Việt Nam hoặc đạt điểm tin cậy cao hơn.
    """
    res_0 = _recognize_single_orientation(cropped_pil, plate_type, geometry_rectified)
    q0 = _recognition_quality(res_0, plate_type)
    if q0 > .98 and not res_0.get('needs_review',True):
        result = res_0
    else:
        res_180 = _recognize_single_orientation(cropped_pil.rotate(180,expand=True),plate_type,geometry_rectified)
        q180 = _recognition_quality(res_180, plate_type)
        result = res_180 if q180 > q0+.03 else res_0
    # Keep layout formatting with spaces replacing '-' for clear readability
    result = dict(result)
    result['plate_text'] = format_plate_display(result.get('plate_text', 'KXD'), plate_type)
    return result


def recognize_license_plate(cropped_pil, plate_type="double_row", geometry_rectified=False, backend=None):
    """Select an explicit reader; annotation/training never happens at prediction."""
    import config
    selected=backend or getattr(config,'OCR_BACKEND','template')
    if selected=='template':
        return recognize_license_plate_template(cropped_pil,plate_type,geometry_rectified)
    if selected!='crnn':
        raise ValueError('Unknown OCR backend: '+str(selected))
    if isinstance(cropped_pil,np.ndarray):cropped_pil=Image.fromarray(cropped_pil)
    try:
        from learning.line_model import recognize
        return recognize(cropped_pil,plate_type)
    except (ImportError,FileNotFoundError) as exc:
        raise RuntimeError('Chưa sẵn sàng bộ đọc học sâu. Chạy tools/setup_learning.ps1 hoặc chọn bộ đọc Truyền thống.') from exc
