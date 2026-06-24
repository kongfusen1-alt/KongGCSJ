import sys, os, json, base64, functools
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2, numpy as np, torch, time
import pymysql
from torchvision.ops import nms as tnms
from flask import (Flask, render_template, Response, jsonify,
                   request, redirect, url_for, session)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import onnxruntime as ort
from config import (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD,
                    DB_NAME, SECRET_KEY)

# ─── Flask ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["SESSION_TYPE"] = "filesystem"
CORS(app)

BASE = os.path.dirname(os.path.dirname(__file__))

# ─── 数据库工具 ─────────────────────────────────────────
def get_db():
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, database=DB_NAME,
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor
    )

def db_query(sql, args=None, fetch=True):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(sql, args)
    result = cur.fetchall() if fetch else None
    conn.commit()
    cur.close()
    conn.close()
    return result

def db_insert(sql, args=None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(sql, args)
    conn.commit()
    lid = cur.lastrowid
    cur.close()
    conn.close()
    return lid

# ─── 登录装饰器 ────────────────────────────────────────
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            return "权限不足，仅管理员可访问", 403
        return f(*args, **kwargs)
    return decorated

# ─── 人脸识别模型 ──────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# YOLOv5-face 人脸检测 — ONNX Runtime
yolo = ort.InferenceSession(
    os.path.join(BASE, "yolov5m-face.onnx"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# ArcFace 特征提取 — ONNX Runtime（省掉 PyTorch 模型显存）
backbone_session = ort.InferenceSession(
    os.path.join(BASE, "backbone.onnx"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
# 缓存 input/output 名称，避免每次推理查
_BACKBONE_IN_NAME = backbone_session.get_inputs()[0].name
_BACKBONE_OUT_NAME = backbone_session.get_outputs()[0].name

# ─── 人脸数据库（内存缓存）─────────────────────────────
def load_face_db():
    global db_feats, db_names, db_ids
    db_feats = None; db_names = []; db_ids = []
    rows = db_query("SELECT id, name, feature_path FROM face_db ORDER BY id")
    for row in rows:
        fp = row["feature_path"]
        if fp and os.path.exists(fp):
            feat = np.load(fp).reshape(1, -1)
            db_names.append(row["name"])
            db_ids.append(row["id"])
            if db_feats is None:
                db_feats = torch.from_numpy(feat).float().to(device)
            else:
                db_feats = torch.cat([db_feats, torch.from_numpy(feat).float().to(device)])
    tensor_dir = os.path.join(BASE, "database_tensor")
    if os.path.exists(tensor_dir):
        for f in sorted(os.listdir(tensor_dir)):
            if f.endswith(".npy"):
                name = f.replace(".npy", "")
                if name not in db_names:
                    feat = np.load(os.path.join(tensor_dir, f)).reshape(1, -1)
                    feat_norm = feat / np.linalg.norm(feat)
                    img_path = os.path.join(BASE, "database_image", f"{name}.jpg")
                    if not os.path.exists(img_path):
                        img_path = None
                    fid = db_insert(
                        "INSERT INTO face_db (name, image_path, feature_path) VALUES (%s,%s,%s)",
                        (name, img_path, os.path.join(tensor_dir, f))
                    )
                    db_names.append(name)
                    db_ids.append(fid)
                    if db_feats is None:
                        db_feats = torch.from_numpy(feat_norm).float().to(device)
                    else:
                        db_feats = torch.cat([db_feats, torch.from_numpy(feat_norm).float().to(device)])
    print(f"Face DB: {len(db_names)} faces loaded")

db_feats = None; db_names = []; db_ids = []
load_face_db()

# ─── 统计 ──────────────────────────────────────────────
current_fps = 0; current_faces = 0
current_frame = None  # 最新一帧，供智能体画面分析用

# ─── 工具函数 ──────────────────────────────────────────
def compute_iou(box1, box2):
    """两个 xyxy 框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def letterbox(im, ns=(640, 640)):
    s = im.shape[:2]
    r = min(ns[0] / s[0], ns[1] / s[1])
    nu = (int(round(s[1] * r)), int(round(s[0] * r)))
    dw, dh = (ns[1] - nu[0]) / 2, (ns[0] - nu[1]) / 2
    if s[::-1] != nu:
        im = cv2.resize(im, nu, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, int(round(dh - 0.1)), int(round(dh + 0.1)),
                            int(round(dw - 0.1)), int(round(dw + 0.1)),
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (dw, dh)

def do_nms(pred, ct=0.5, it=0.45):
    xc = pred[..., 4] > ct
    out = [torch.zeros((0, 16))] * pred.shape[0]
    for xi, x in enumerate(pred):
        x = x[xc[xi]]
        if not x.shape[0]: continue
        x[:, 15:] *= x[:, 4:5]
        bx = torch.zeros_like(x[:, :4])
        bx[:, 0] = x[:, 0] - x[:, 2] / 2
        bx[:, 1] = x[:, 1] - x[:, 3] / 2
        bx[:, 2] = x[:, 0] + x[:, 2] / 2
        bx[:, 3] = x[:, 1] + x[:, 3] / 2
        cf, j = x[:, 15:].max(1, keepdim=True)
        x = torch.cat((bx, cf, x[:, 5:15], j.float()), 1)[cf.view(-1) > ct]
        if x.shape[0]:
            k = tnms(x[:, :4] + x[:, 15:16] * 4096, x[:, 4], it)
            x = x[k]
        out[xi] = x
    return out

# ─── 人脸对齐 — ArcFace 标准 5 点模板 (112×112) ────────
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # 左眼
    [73.5318, 51.5014],   # 右眼
    [56.0252, 71.7366],   # 鼻子
    [41.5493, 92.3655],   # 左嘴角
    [70.7299, 92.2041],   # 右嘴角
], dtype=np.float32)


def _extract_landmarks(raw_box, dw, dh, r, w, h):
    """从 YOLO 原始输出中提取 5 个关键点并转换到原图坐标"""
    landmarks = np.zeros((5, 2), dtype=np.float32)
    for i in range(5):
        lx = (raw_box[5 + i * 2] - dw) / r
        ly = (raw_box[6 + i * 2] - dh) / r
        landmarks[i] = [max(0, min(w, lx)), max(0, min(h, ly))]
    return landmarks


def _detect_face(frame):
    """YOLOv5-face 检测，返回 (face_crop, x0, y0, x1, y1, landmarks_orig) 或 None"""
    h, w = frame.shape[:2]
    r = 640 / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    img_resized = cv2.resize(frame, (new_w, new_h))
    dw, dh = (640 - new_w) // 2, (640 - new_h) // 2
    img_padded = cv2.copyMakeBorder(
        img_resized, dh, 640 - new_h - dh, dw, 640 - new_w - dw,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    img_input = img_padded.transpose((2, 0, 1))
    img_input = np.expand_dims(img_input, 0).astype(np.float32) / 255.0
    outputs = yolo.run(["output"], {"input": img_input})[0]

    boxes = outputs[0]
    confs = boxes[:, 4]
    if np.max(confs) <= 0.5:
        return None

    idx = int(np.argmax(confs))
    best = boxes[idx]
    cx, cy, bw, bh = best[:4]
    x0 = int((cx - bw / 2 - dw) / r)
    y0 = int((cy - bh / 2 - dh) / r)
    x1 = int((cx + bw / 2 - dw) / r)
    y1 = int((cy + bh / 2 - dh) / r)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    face_roi = frame[y0:y1, x0:x1]
    if face_roi.size == 0 or face_roi.shape[0] < 30 or face_roi.shape[1] < 30:
        return None
    landmarks = _extract_landmarks(best, dw, dh, r, w, h)
    return face_roi, x0, y0, x1, y1, landmarks


def _align_face(face_crop, landmarks, x0, y0):
    """5 点相似变换对齐人脸到 ArcFace 标准 112×112"""
    crop_lm = landmarks - np.array([x0, y0], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(crop_lm, REFERENCE_LANDMARKS)
    if M is None:
        return cv2.resize(face_crop, (112, 112))
    return cv2.warpAffine(face_crop, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)


# ─── 活体检测：YCrCb 色彩分析 + 纹理分析 ────────────────
# 无需外部模型，纯 OpenCV 实现


# ─── 活体检测：MiniFASNet-V2 (ONNX) ─────────────────────
MINIFASNET_PATH = os.path.join(BASE, "minifasnet.onnx")
if os.path.exists(MINIFASNET_PATH):
    minifasnet_session = ort.InferenceSession(
        MINIFASNET_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    _MF_INPUT_NAME = minifasnet_session.get_inputs()[0].name
    _MF_OUTPUT_NAME = minifasnet_session.get_outputs()[0].name
    _MF_H, _MF_W = minifasnet_session.get_inputs()[0].shape[2:]  # 80,80
    print(f"[Liveness] MiniFASNet 已加载 ({_MF_W}x{_MF_H})")
else:
    minifasnet_session = None
    print(f"[WARN] 未找到 {MINIFASNET_PATH}，活体检测禁用")


def _crop_expanded(bgr, box_xyxy, scale=2.7):
    """从原图按 scale 放大裁剪人脸区域（原版 MiniFASNet 方法）
    box_xyxy: [x0, y0, x1, y1]
    返回放大后的 crop
    """
    src_h, src_w = bgr.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    box_w = x1 - x0
    box_h = y1 - y0
    cx = x0 + box_w / 2
    cy = y0 + box_h / 2

    s = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w = box_w * s
    new_h = box_h * s

    lx = max(0, int(cx - new_w / 2))
    ly = max(0, int(cy - new_h / 2))
    rx = min(src_w - 1, int(cx + new_w / 2))
    ry = min(src_h - 1, int(cy + new_h / 2))
    return bgr[ly:ry + 1, lx:rx + 1]


def _liveness_score(face_bgr):
    """MiniFASNet 推理。face_bgr 是已放大的人脸 crop，BGR 0-255。"""
    if minifasnet_session is None or face_bgr.size == 0:
        return 0.5
    try:
        face = cv2.resize(face_bgr, (_MF_W, _MF_H))
        face = face.astype(np.float32)  # 保持 0-255，不做归一化
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, 0)
        logits = minifasnet_session.run([_MF_OUTPUT_NAME], {_MF_INPUT_NAME: face})[0]
        # Softmax
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        score = float(probs[0, 1])  # class 1 = Real
        print(f"[Liveness] score={score:.3f}  probs={probs[0]}")
        return score
    except Exception as e:
        print(f"[Liveness] 异常: {e}")
        return 0.5


def is_blurry(img, threshold=80):
    """拉普拉斯方差 < threshold 判定为模糊，返回 (bool, var)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var


def rec(cropped, landmarks=None, x0=0, y0=0):
    """人脸识别，支持传入 landmarks 做对齐。landmarks 为原图坐标。"""
    global db_feats, db_names
    if db_feats is None:
        return "face"
    try:
        if cropped.shape[0] < 30 or cropped.shape[1] < 30:
            return "face"
        # 5 点对齐（如有 landmarks）
        if landmarks is not None:
            face = _align_face(cropped, landmarks, x0, y0)
        else:
            face = cv2.resize(cropped, (112, 112))
        f = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        f = np.transpose(f, (2, 0, 1))
        f = np.expand_dims(f, 0).astype(np.float32) / 255.0
        f = (f - 0.5) / 0.5
        # ONNX Runtime 推理 backbone.onnx
        feat = backbone_session.run([_BACKBONE_OUT_NAME], {_BACKBONE_IN_NAME: f})[0]
        feat = feat / np.linalg.norm(feat)
        # 转 torch tensor 在 GPU 上做距离计算
        feat_t = torch.from_numpy(feat).to(device)
        ds = torch.norm(db_feats - feat_t, dim=1)
        d, idx = torch.min(ds, dim=0)
        dist = d.item()
        name = db_names[idx] if dist < 1.5 else "unknown"
        return name
    except Exception as e:
        print(f"[rec] 异常: {e}")
        return "face"

# ─── 视频流 ────────────────────────────────────────────
import threading

class CameraManager:
    """带引用计数的摄像头管理器，线程安全"""
    def __init__(self):
        self._cam = None
        self._lock = threading.Lock()
        self._refcount = 0

    def acquire(self):
        with self._lock:
            if self._cam is None or not self._cam.isOpened():
                if self._cam is not None:
                    self._cam.release()
                self._cam = cv2.VideoCapture(0)
                if not self._cam.isOpened():
                    for i in range(1, 5):
                        self._cam = cv2.VideoCapture(i)
                        if self._cam.isOpened():
                            break
                if self._cam and self._cam.isOpened():
                    print("CAMERA OK")
                else:
                    print("NO CAMERA")
                    return False
            self._refcount += 1
            return True

    def release(self):
        with self._lock:
            self._refcount = max(0, self._refcount - 1)
            if self._refcount == 0 and self._cam is not None:
                self._cam.release()
                self._cam = None
                print("CAMERA released (refcount=0)")

    @property
    def is_opened(self):
        return self._cam is not None and self._cam.isOpened()

    def read(self):
        with self._lock:
            if self._cam is None or not self._cam.isOpened():
                return False, None
            return self._cam.read()

cam_mgr = CameraManager()
last_detections = []  # 缓存上次检测结果

# 自动日志：每人每 N 秒最多记一条
_LIVENESS_THRESHOLD = 0.5  # MiniFASNet 活体概率阈值，>= 为真人


# ─── 日志：记录人员出现/离开事件 ────────────────────────
_face_presence = {}  # name -> last_seen_time
_PRESENCE_TIMEOUT = 3.0  # 秒，超过此时间未出现视为离开

def _log_event(name, event_type, confidence=0):
    try:
        db_insert(
            "INSERT INTO face_recognition_log (name, event_type, confidence) VALUES (%s,%s,%s)",
            (name, event_type, float(confidence))
        )
        print(f"[Log] {event_type}: {name} (conf={confidence})")
    except Exception as e:
        print(f"[Log] 写入失败: {e}")

def _log_detections(detections):
    """记录出现/离开事件"""
    global _face_presence
    now = time.time()
    detected_names = set()

    for item in detections:
        nm = item[1]
        sc = item[2]
        if nm in ("unknown", "face", "blurry"):
            continue
        detected_names.add(nm)
        if nm not in _face_presence:
            _log_event(nm, "appeared", sc)
        _face_presence[nm] = now

    # 检查离开：之前出现但当前帧未检测到的人
    gone = [nm for nm, t in _face_presence.items()
            if nm not in detected_names and now - t > _PRESENCE_TIMEOUT]
    for nm in gone:
        _log_event(nm, "disappeared")
        del _face_presence[nm]

def gen():
    global current_fps, current_faces, last_detections, current_frame
    if not cam_mgr.acquire():
        return
    fvs = []
    frame_skip = 3
    frame_count = 0
    try:
        while True:
            ret, frame = cam_mgr.read()
            if not ret:
                # 等待 200ms 后重试，而不是直接退出
                time.sleep(0.2)
                continue
            t1 = time.time()

            frame_count += 1
            if frame_count % frame_skip == 1:
                img, ratio, dwdh = letterbox(frame)
                im = img.transpose((2, 0, 1))
                im = np.expand_dims(im, 0).astype(np.float32) / 255.0
                ops = yolo.run(["output"], {"input": im})[0]
                out = do_nms(torch.from_numpy(ops))[0]
                last_detections = []
                all_boxes = []
                for det in out:
                    x0, y0, x1, y1, sc = det[:5].tolist()
                    bx = np.array([x0, y0, x1, y1])
                    bx -= np.array(dwdh * 2)
                    bx /= ratio
                    bx = bx.round().astype(np.int32).tolist()
                    cr = frame[max(0, bx[1]):min(frame.shape[0], bx[3]),
                               max(0, bx[0]):min(frame.shape[1], bx[2])]
                    if cr.shape[0] < 50 or cr.shape[1] < 50: continue
                    # 提取 landmarks（从 letterbox 坐标转到原图坐标）
                    lm_raw = det[5:15].tolist()
                    lm = np.array(lm_raw, dtype=np.float32).reshape(5, 2)
                    lm[:, 0] = (lm[:, 0] - dwdh[0]) / ratio
                    lm[:, 1] = (lm[:, 1] - dwdh[1]) / ratio
                    # 模糊检测：太模糊则跳过 ArcFace 推理
                    blurry, lv = is_blurry(cr, threshold=80)
                    # ═══ 活体检测：MiniFASNet ═══
                    # 从原图按 2.7x 放大裁剪（原版模型要求的输入）
                    face_ctx = _crop_expanded(frame, bx)
                    liveness = _liveness_score(face_ctx)
                    if blurry:
                        nm = "blurry"
                    else:
                        nm = rec(cr, landmarks=lm, x0=bx[0], y0=bx[1])
                    last_detections.append((bx, nm, sc, liveness))

                # 自动记录识别日志
                _log_detections(last_detections)

                # FPS 只在实际执行检测的帧上计算
                dt = max(time.time() - t1, 1e-6)
                fps = 1.0 / dt
                fvs.append(fps)
                if len(fvs) > 30: fvs.pop(0)
                current_fps = round(sum(fvs) / len(fvs), 1)
                # 只在检测帧更新快照（减少 memory copy）
                current_frame = frame.copy()

            fc = 0
            for bx, nm, sc, liveness in last_detections:
                fc += 1
                is_live = liveness >= _LIVENESS_THRESHOLD
                color = (0, 255, 0) if is_live else (0, 0, 255)
                label = f"{nm} {sc:.2f}"
                label += " [OK]" if is_live else " [!!]"
                cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
                # 标签背景
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (bx[0], bx[1] - th - 8), (bx[0] + tw + 6, bx[1]), color, -1)
                cv2.putText(frame, label, (bx[0] + 3, bx[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            current_faces = fc
            cv2.putText(frame, f"FPS:{current_fps} Faces:{fc}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            r, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    except GeneratorExit:
        pass
    finally:
        cam_mgr.release()


# ═══════════════════════════════════════════════════════
#  路由
# ═══════════════════════════════════════════════════════

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if len(username) < 2 or len(password) < 4:
        return jsonify({"error": "用户名至少2位，密码至少4位"}), 400
    exist = db_query("SELECT id FROM users WHERE username=%s", (username,))
    if exist:
        return jsonify({"error": "用户名已存在"}), 400
    pwd_hash = generate_password_hash(password)
    uid = db_insert("INSERT INTO users (username, password, role) VALUES (%s,%s,%s)",
                    (username, pwd_hash, "user"))
    return jsonify({"success": True, "id": uid})


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if "user_id" in session:
            return redirect(url_for("dashboard"))
        return render_template("login.html")
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not username or not password:
        return render_template("login.html", error="请输入用户名和密码")
    user = db_query("SELECT * FROM users WHERE username=%s", (username,))
    if not user or not check_password_hash(user[0]["password"], password):
        return render_template("login.html", error="用户名或密码错误")
    session["user_id"] = user[0]["id"]
    session["username"] = user[0]["username"]
    session["role"] = user[0]["role"]
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─── 功能实施页 ────────────────────────────────────────
@app.route("/")
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html",
                           names=db_names,
                           username=session["username"],
                           role=session["role"])

# ─── 用户管理 ──────────────────────────────────────────
@app.route("/user_manage")
@login_required
def user_manage():
    if session["role"] != "admin":
        return "权限不足", 403
    users = db_query("SELECT id, username, role, created_at FROM users ORDER BY id")
    return render_template("user_manage.html",
                           users=users,
                           username=session["username"],
                           role=session["role"])

@app.route("/api/users", methods=["GET"])
@admin_required
def api_get_users():
    users = db_query("SELECT id, username, role, created_at FROM users ORDER BY id")
    return jsonify(users)

@app.route("/api/users", methods=["POST"])
@admin_required
def api_add_user():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "")
    role = data.get("role", "user")
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if role not in ("admin", "user"):
        return jsonify({"error": "角色无效"}), 400
    exist = db_query("SELECT id FROM users WHERE username=%s", (username,))
    if exist:
        return jsonify({"error": "用户名已存在"}), 400
    pwd_hash = generate_password_hash(password)
    uid = db_insert("INSERT INTO users (username, password, role) VALUES (%s,%s,%s)",
                    (username, pwd_hash, role))
    return jsonify({"success": True, "id": uid})

@app.route("/api/users/<int:uid>", methods=["PUT"])
@admin_required
def api_edit_user(uid):
    data = request.get_json()
    role = data.get("role")
    password = data.get("password", "")
    if role and role in ("admin", "user"):
        if password:
            pwd_hash = generate_password_hash(password)
            db_query("UPDATE users SET role=%s, password=%s WHERE id=%s",
                     (role, pwd_hash, uid), fetch=False)
        else:
            db_query("UPDATE users SET role=%s WHERE id=%s", (role, uid), fetch=False)
    return jsonify({"success": True})

@app.route("/api/users/<int:uid>", methods=["DELETE"])
@admin_required
def api_delete_user(uid):
    if uid == session.get("user_id"):
        return jsonify({"error": "不能删除自己"}), 400
    db_query("DELETE FROM users WHERE id=%s", (uid), fetch=False)
    return jsonify({"success": True})

# ─── 人脸管理 ──────────────────────────────────────────
@app.route("/face_manage")
@login_required
def face_manage():
    faces = db_query("SELECT id, name, image_path, created_at FROM face_db ORDER BY id")
    return render_template("face_manage.html",
                           faces=faces,
                           username=session["username"],
                           role=session["role"])

@app.route("/api/faces/preview", methods=["POST"])
@login_required
def api_preview_face():
    """拍照并检测人脸，返回裁剪后的人脸预览图（不保存）"""
    if not cam_mgr.acquire():
        return jsonify({"error": "摄像头未打开"}), 400
    ret, frame = cam_mgr.read()
    cam_mgr.release()
    if not ret:
        return jsonify({"error": "拍照失败"}), 400

    result = _detect_face(frame)
    if result is None:
        return jsonify({"error": "未检测到人脸，请面对摄像头"}), 400
    face_roi, x0, y0, x1, y1, landmarks = result
    # 对齐后预览
    aligned = _align_face(face_roi, landmarks, x0, y0)
    _, buf = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf.tobytes()).decode()
    return jsonify({
        "success": True,
        "image": f"data:image/jpeg;base64,{img_b64}",
        "bbox": [x0, y0, x1, y1]
    })


@app.route("/api/faces", methods=["GET"])
@login_required
def api_get_faces():
    faces = db_query("SELECT id, name, image_path, created_at FROM face_db ORDER BY id")
    return jsonify([{"id": f["id"], "name": f["name"],
                     "has_image": f["image_path"] is not None} for f in faces])

@app.route("/api/faces", methods=["POST"])
@login_required
def api_add_face():
    global db_feats, db_names, db_ids
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "请输入名字"}), 400

    # 接受预览时传回的 base64 图片（已对齐），不用重新拍照
    image_data = data.get("image_data", "")
    if image_data:
        header, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
        img_bytes = base64.b64decode(encoded)
        img_arr = np.frombuffer(img_bytes, np.uint8)
        face_aligned = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if face_aligned is None or face_aligned.size == 0:
            return jsonify({"error": "图片解码失败"}), 400
    else:
        # 兼容旧调用（直接拍照 + 对齐）
        if not cam_mgr.acquire():
            return jsonify({"error": "摄像头未打开"}), 400
        ret, frame = cam_mgr.read()
        cam_mgr.release()
        if not ret:
            return jsonify({"error": "拍照失败"}), 400
        result = _detect_face(frame)
        if result is None:
            return jsonify({"error": "未检测到人脸，请面对摄像头"}), 400
        face_bgr, x0, y0, x1, y1, landmarks = result
        face_aligned = _align_face(face_bgr, landmarks, x0, y0)

    img_dir = os.path.join(BASE, "database_image")
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"{name}.jpg")
    cv2.imwrite(img_path, face_aligned)

    f = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
    f = np.transpose(f, (2, 0, 1))
    f = np.expand_dims(f, 0).astype(np.float32) / 255.0
    f = (f - 0.5) / 0.5
    feat = backbone_session.run([_BACKBONE_OUT_NAME], {_BACKBONE_IN_NAME: f})[0]
    feat = feat / np.linalg.norm(feat)
    feat_dir = os.path.join(BASE, "database_tensor")
    os.makedirs(feat_dir, exist_ok=True)
    feat_path = os.path.join(feat_dir, f"{name}.npy")
    np.save(feat_path, feat)
    fid = db_insert(
        "INSERT INTO face_db (name, image_path, feature_path) VALUES (%s,%s,%s)",
        (name, img_path, feat_path)
    )
    db_names.append(name)
    db_ids.append(fid)
    feat_t = torch.from_numpy(feat).float().to(device)
    if db_feats is None:
        db_feats = feat_t
    else:
        db_feats = torch.cat([db_feats, feat_t])
    return jsonify({"success": True, "name": name, "id": fid})

@app.route("/api/faces/<int:fid>", methods=["DELETE"])
@login_required
def api_delete_face(fid):
    global db_feats, db_names, db_ids
    row = db_query("SELECT name, image_path, feature_path FROM face_db WHERE id=%s", (fid,))
    if not row:
        return jsonify({"error": "记录不存在"}), 404
    r = row[0]
    if r["image_path"] and os.path.exists(r["image_path"]):
        os.remove(r["image_path"])
    if r["feature_path"] and os.path.exists(r["feature_path"]):
        os.remove(r["feature_path"])
    db_query("DELETE FROM face_db WHERE id=%s", (fid), fetch=False)
    # 直接从内存移除，不用重载整个库
    if fid in db_ids:
        idx = db_ids.index(fid)
        db_names.pop(idx)
        db_ids.pop(idx)
        if db_feats is not None and db_feats.shape[0] > 0:
            keep = torch.ones(db_feats.shape[0], dtype=torch.bool, device=device)
            keep[idx] = False
            db_feats = db_feats[keep]
            if db_feats.shape[0] == 0:
                db_feats = None
    return jsonify({"success": True})

@app.route("/api/face_image/<int:fid>")
@login_required
def api_face_image(fid):
    row = db_query("SELECT image_path FROM face_db WHERE id=%s", (fid,))
    if row and row[0]["image_path"] and os.path.exists(row[0]["image_path"]):
        from flask import send_file
        return send_file(row[0]["image_path"], mimetype="image/jpeg")
    return "", 404

# ─── 视频流 & 统计 ────────────────────────────────────
@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stats")
@login_required
def get_stats():
    return jsonify({"fps": current_fps, "faces": current_faces})

@app.route("/api/names")
@login_required
def get_names():
    return jsonify({"names": db_names})

# ─── 手动保存识别日志 ──────────────────────────────────
@app.route("/api/log/save", methods=["POST"])
@login_required
def api_log_save():
    """手动保存当前识别结果到日志"""
    if not last_detections:
        return jsonify({"message": "当前没有识别结果"})
    count = 0
    for item in last_detections:
        bx, nm, sc = item[0], item[1], item[2]
        evt = "appeared" if nm not in ("unknown", "face") else "unknown"
        db_insert(
            "INSERT INTO face_recognition_log (name, event_type, confidence) VALUES (%s,%s,%s)",
            (nm, evt, float(sc))
        )
        count += 1
    return jsonify({"success": True, "saved": count})

# ─── 智能体路由 ────────────────────────────────────────
from agent.llm_core import FaceAgent
face_agent = FaceAgent(db_query, db_insert,
                       lambda: {"fps": current_fps, "faces": current_faces, "db_faces": len(db_names)})

@app.route("/agent")
@login_required
def agent_page():
    return render_template("agent_chat.html",
                           username=session["username"],
                           role=session["role"])

@app.route("/api/agent/chat", methods=["POST"])
@login_required
def agent_chat():
    data = request.get_json()
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "消息不能为空"}), 400
    reply = face_agent.chat(msg, frame=current_frame)
    return jsonify({"reply": reply})

@app.route("/api/agent/recent_log")
@login_required
def agent_recent_log():
    rows = db_query(
        "SELECT name, event_type, confidence, created_at FROM face_recognition_log ORDER BY created_at DESC LIMIT 100"
    )
    records = []
    for r in (rows or []):
        records.append({
            "name": r["name"],
            "event_type": r["event_type"],
            "time": r["created_at"].strftime("%H:%M:%S") if r["created_at"] else None
        })
    return jsonify({"records": records})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
