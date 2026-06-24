"""
拍照入库 — 带 YOLO 人脸检测
打开摄像头，拍照后先 YOLO 检测人脸再提取特征，保证特征质量
"""
import cv2, os, numpy as np, torch
import onnxruntime as ort
from iresnet import iresnet100

# ─── YOLO 检测 ───
yolo_session = ort.InferenceSession("yolov5m-face.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

def detect_largest_face(img_orig):
    h, w = img_orig.shape[:2]
    r = 640 / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    img_resized = cv2.resize(img_orig, (new_w, new_h))
    dw, dh = (640 - new_w) // 2, (640 - new_h) // 2
    img_padded = cv2.copyMakeBorder(img_resized, dh, 640-new_h-dh, dw, 640-new_w-dw, cv2.BORDER_CONSTANT, value=(114,114,114))
    img_input = img_padded.transpose((2,0,1))
    img_input = np.expand_dims(img_input, 0).astype(np.float32) / 255.0
    outputs = yolo_session.run(["output"], {"input": img_input})[0]
    boxes = outputs[0]
    confs = boxes[:, 4]
    if np.max(confs) > 0.5:
        best = boxes[np.argmax(confs)]
        cx, cy, bw, bh = best[:4]
        x0 = int((cx - bw/2 - dw) / r)
        y0 = int((cy - bh/2 - dh) / r)
        x1 = int((cx + bw/2 - dw) / r)
        y1 = int((cy + bh/2 - dh) / r)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        face_roi = img_orig[y0:y1, x0:x1]
        if face_roi.size > 0 and face_roi.shape[0] > 20 and face_roi.shape[1] > 20:
            return cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

# ─── 特征提取 ───
def extract_feature(face_rgb):
    face = cv2.resize(face_rgb, (112, 112))
    face = np.transpose(face, (2,0,1))
    face = np.expand_dims(face, 0).astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    with torch.no_grad():
        feat = net(torch.from_numpy(face).cuda()).cpu().numpy()
    return feat / np.linalg.norm(feat)

# ─── 主流程 ───
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

print("面对摄像头，按 SPACE 拍照，按 ESC 取消")
captured = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, "Press SPACE to capture, ESC to cancel", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        captured = frame.copy()
        cv2.putText(captured, "Captured!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture", captured)
        cv2.waitKey(500)
        break

cap.release()
cv2.destroyAllWindows()

if captured is None:
    print("未拍照")
    exit()

# 保存原图
name = input("输入姓名（不含扩展名）: ").strip()
if not name:
    name = "new_face"
img_path = os.path.join("database_image", f"{name}.jpg")
cv2.imwrite(img_path, captured)
print(f"已保存: {img_path}")

# YOLO 检测人脸
face_rgb = detect_largest_face(captured)

# 加载模型
net = iresnet100(False)
net.load_state_dict(torch.load("backbone.pth", map_location="cpu"))
net.eval().cuda()

# 提取特征（归一化后保存）
feat = extract_feature(face_rgb)
save_path = os.path.join("database_tensor", f"{name}.npy")
np.save(save_path, feat)
print(f"特征已保存: {save_path}")
print(f"  特征维度: {feat.shape[1]}")
print("✅ 完成！新面孔已加入特征库")
