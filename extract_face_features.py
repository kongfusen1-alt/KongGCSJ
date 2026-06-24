import os, cv2, numpy as np, torch
import onnxruntime as ort
from iresnet import iresnet100

yolo_session = ort.InferenceSession("yolov5m-face.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

net = iresnet100(False)
net.load_state_dict(torch.load("backbone.pth", map_location="cpu"))
net.eval().cuda()

os.makedirs("database_tensor", exist_ok=True)

for im in sorted(os.listdir("database_image")):
    if not im.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_orig = cv2.imread(os.path.join("database_image", im))
    if img_orig is None:
        continue

    # 用 YOLO 检测人脸
    h, w = img_orig.shape[:2]
    r = 640 / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    img_resized = cv2.resize(img_orig, (new_w, new_h))
    dw, dh = (640 - new_w) // 2, (640 - new_h) // 2
    img_padded = cv2.copyMakeBorder(img_resized, dh, 640-new_h-dh, dw, 640-new_w-dw, cv2.BORDER_CONSTANT, value=(114,114,114))

    img_input = img_padded.transpose((2,0,1))
    img_input = np.expand_dims(img_input, 0).astype(np.float32) / 255.0

    outputs = yolo_session.run(["output"], {"input": img_input})[0]

    name_orig = im.rsplit(".", 1)[0]

    # 取最高置信度的人脸框
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
        if face_roi.size > 0 and face_roi.shape[0] > 10 and face_roi.shape[1] > 10:
            face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        else:
            face = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    else:
        face = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    face = cv2.resize(face, (112, 112))
    face = np.transpose(face, (2,0,1))
    face = np.expand_dims(face, 0).astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5

    with torch.no_grad():
        feat = net(torch.from_numpy(face).cuda()).cpu().numpy()

    # ✅ L2 归一化后保存
    feat = feat / np.linalg.norm(feat)

    save_path = os.path.join("database_tensor", name_orig + ".npy")
    np.save(save_path, feat)
    print(f"  {im} -> OK (归一化)")

print("特征库已更新！所有特征已 L2 归一化")
