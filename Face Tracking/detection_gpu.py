"""
FaceID: YOLOv5 Face Detection + ArcFace Recognition (GPU)
"""
import random
import numpy as np
import cv2
import time
import torch
import torchvision
import os


class FaceDetector:
    def __init__(self, img_path, yolo_model, backbone_model, conf_thres,
                 iou_thres, img_size, classes, webcam=False):
        self.webcam = webcam
        self.img_path = img_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.names = classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_count = 0
        self.last_names = {}

        import onnxruntime as ort
        self.ort_session = ort.InferenceSession(
            yolo_model,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print(f"[INFO] YOLO 检测: ONNX Runtime")

        from iresnet import iresnet100
        self.backbone = iresnet100(False)
        self.backbone.load_state_dict(torch.load(backbone_model, map_location='cpu'))
        self.backbone.eval()
        self.backbone = self.backbone.to(self.device)
        print(f"[INFO] 人脸识别: {self.device} (PyTorch)")

        # 加载特征库（兼容未归一化的旧特征）
        self.db_features = None
        self.db_names = []
        if os.path.exists('database_tensor'):
            feats = []
            for f in sorted(os.listdir('database_tensor')):
                if f.endswith('.npy'):
                    feat = np.load(os.path.join('database_tensor', f))
                    feat = feat / np.linalg.norm(feat)  # 确保归一化
                    feats.append(feat)
                    self.db_names.append(f.replace('.npy', ''))
            if feats:
                self.db_features = torch.from_numpy(np.vstack(feats)).float().to(self.device)
                print(f"[INFO] 特征库: {len(self.db_names)} 个人")

    def __call__(self):
        if self.webcam:
            self._run_webcam()
        else:
            image_or = cv2.imread(self.img_path)
            if image_or is None:
                print(f"[ERROR] 无法读取图片: {self.img_path}")
                return
            self.detect_img(image_or)

    def _run_webcam(self):
        vid = cv2.VideoCapture(0)
        if not vid.isOpened():
            for i in range(1, 5):
                vid = cv2.VideoCapture(i)
                if vid.isOpened():
                    break
        if not vid.isOpened():
            print("[ERROR] 无法打开摄像头")
            return

        print("[INFO] 摄像头已打开，按 Q 退出，按 S 截图")
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        fps_values = []
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            t1 = time.time()
            output = self.detect_img(frame)
            fps = 1.0 / (time.time() - t1)
            fps_values.append(fps)
            if len(fps_values) > 30:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)

            cv2.putText(output, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output, f"GPU: ON", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('FaceID', output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('screenshot.jpg', output)
                print("[OK] 截图已保存 screenshot.jpg")

        vid.release()
        cv2.destroyAllWindows()

    def detect_img(self, image_or):
        image, ratio, dwdh = self.letterbox(image_or, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255

        session = self.ort_session
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        inp = np.array(im, dtype=np.float32)
        try:
            outputs = session.run(outname, {inname[0]: inp})[0]
        except Exception as e:
            print(f"[ERROR] 推理失败: {e}")
            return image_or

        output = torch.from_numpy(outputs)
        out = self.non_max_suppression_face(output, self.conf_thres, self.iou_thres)[0]
        img = self.result(image_or, ratio, dwdh, out)
        return img

    def result(self, img, ratio, dwdh, out):
        names = self.class_name()
        colors = {name: [random.randint(0, 255) for _ in range(3)]
                  for i, name in enumerate(names)}

        for i, (x0, y0, x1, y1, score) in enumerate(out[:, 0:5]):
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            score = round(float(score), 3)
            name = names[0]
            color = colors[name]

            cropped_box = img[box[1]:box[3], box[0]:box[2]]
            if cropped_box.shape[0] == 0 or cropped_box.shape[1] == 0:
                continue

            box_id = f"{box[0]}_{box[1]}"
            self.frame_count += 1
            # 每~6帧（约0.2秒）才做一次识别，中间帧复用上次结果
            if box_id in self.last_names and self.frame_count % 6 != 0:
                recognized_name = self.last_names[box_id]
            else:
                recognized_name = self._recognize_face(cropped_box)
                self.last_names[box_id] = recognized_name

            cv2.rectangle(img, box[:2], box[2:], color, 2)
            label = f"{recognized_name} {score:.2f}"
            cv2.putText(img, label, (box[0], box[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        [225, 255, 255], thickness=2)
        return img

    @torch.no_grad()
    def _recognize_face(self, cropped_box):
        if self.db_features is None:
            return "face"
        try:
            face = cv2.cvtColor(cropped_box, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (112, 112))
            face = np.transpose(face, (2, 0, 1))
            face = np.expand_dims(face, 0).astype(np.float32)
            face = face / 255.0
            face = (face - 0.5) / 0.5
            face_tensor = torch.from_numpy(face).to(self.device)
            feat = self.backbone(face_tensor)
            feat = feat / torch.norm(feat)
            dists = torch.norm(self.db_features - feat, dim=1)
            best_dist, best_idx = torch.min(dists, dim=0)
            best_dist = best_dist.item()
            if best_dist > 1.5:
                return "unknown"
            return self.db_names[best_idx]
        except Exception as e:
            return "face"

    def box_iou(self, box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def non_max_suppression_face(self, prediction, conf_thres=0.25, iou_thres=0.45,
                                  classes=None, agnostic=False, labels=()):
        nc = prediction.shape[2] - 15
        xc = prediction[..., 4] > conf_thres
        min_wh, max_wh = 2, 4096
        time_limit = 10.0
        redundant = True
        multi_label = nc > 1
        merge = False
        t = time.time()
        output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 15), device=x.device)
                v[:, :4] = l[:, 1:5]
                v[:, 4] = 1.0
                v[range(len(l)), l[:, 0].long() + 15] = 1.0
                x = torch.cat((x, v), 0)
            if not x.shape[0]:
                continue
            x[:, 15:] *= x[:, 4:5]
            box = self.xywh2xyxy(x[:, :4])
            if multi_label:
                i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 15, None], x[i, 5:15], j[:, None].float()), 1)
            else:
                conf, j = x[:, 15:].max(1, keepdim=True)
                x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            n = x.shape[0]
            if not n:
                continue
            c = x[:, 15:16] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            if merge and (1 < n < 3E3):
                iou = self.box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
                if redundant:
                    i = i[iou.sum(1) > 1]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break
        return output

    def xywh2xyxy(self, x):
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def class_name(self):
        classes = []
        with open(self.names, "r") as file:
            while True:
                name = file.readline().strip("\n")
                classes.append(name)
                if not name:
                    break
        return classes

    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        shape = im.shape[:2]
        new_shape = self.img_size
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FaceID: YOLOv5 + ArcFace 人脸检测识别")
    parser.add_argument("--image", type=str, default=None, help="图片路径")
    parser.add_argument("--yolo", type=str, default="./yolov5m-face.onnx", help="YOLOv5 ONNX模型")
    parser.add_argument("--backbone", type=str, default="backbone.pth", help="ArcFace PyTorch权重")
    parser.add_argument("--conf", type=float, default=0.7, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU阈值")
    parser.add_argument("--img_size", type=int, default=640, help="输入大小")
    parser.add_argument("--classes", type=str, default=".\\yolov5-face\\classes.txt", help="类别文件")
    parser.add_argument("--no_webcam", action="store_true", help="不使用摄像头")
    args = parser.parse_args()

    webcam_mode = not args.no_webcam
    img_path = args.image
    if img_path:
        webcam_mode = False
    else:
        img_path = ".\\yolov5-face\\data\\images\\test.jpg"

    print("=" * 50)
    print("   FaceID 人脸检测识别系统")
    print("=" * 50)
    print(f"  YOLO检测: ONNX")
    print(f"  人脸识别: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')} (PyTorch)")
    print(f"  模式: {'摄像头' if webcam_mode else '图片'}")
    print(f"  置信度: {args.conf}  IOU: {args.iou}")
    db_count = len(os.listdir('database_tensor')) if os.path.exists('database_tensor') else 0
    print(f"  特征库: {db_count} 个人脸")
    print("=" * 50)

    det = FaceDetector(
        img_path, args.yolo, args.backbone, args.conf, args.iou,
        (args.img_size, args.img_size), classes=args.classes, webcam=webcam_mode
    )
    det()
