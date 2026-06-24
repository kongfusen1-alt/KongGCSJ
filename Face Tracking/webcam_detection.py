"""
旧版摄像头人脸检测（实验性脚本）
注意：此脚本已在 detection_gpu.py 中重构，此处仅为向后兼容保留。

存在问题（已在新版修复）：
  1. 使用了 iresnet50，但 backbone.pth 是 iresnet100 的权重 → 崩溃
  2. 人脸比对距离计算逻辑混乱
  3. 重复导入、重复方法定义

推荐使用 detection_gpu.py 或 web_ui/app.py
"""
import onnxruntime as ort
import random
import numpy as np
import cv2
import time
import torch
import torchvision
import os
from iresnet import iresnet100  # 修正：使用 iresnet100
from pathlib import Path


class ort_v5:
    def __init__(self, img_path, onnx_model, conf_thres, iou_thres, img_size, classes, webcam=False):
        self.webcam = webcam
        self.img_path = img_path
        self.onnx_model = onnx_model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.names = classes
        self.net = None
        self.ort_session = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(self.onnx_model, providers=providers)
        self.net = iresnet100(False)  # 修正：使用 iresnet100
        self.net.load_state_dict(torch.load('backbone.pth', map_location=torch.device('cpu')))
        self.net = self.net.to(self.device)
        self.net.eval()

        if self.webcam:
            vid = cv2.VideoCapture(0)
            cnt = 0
            while True:
                ret, frame = vid.read()
                cnt += 1
                if cnt % 1 != 0:
                    continue
                t_1 = time.time()
                output = self.detect_img(frame)
                t_2 = time.time()
                print(f"Inference: {t_2 - t_1:.3f}s")
                cv2.imshow('Face Detection', output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            vid.release()
            cv2.destroyAllWindows()
        else:
            image_or = cv2.imread(self.img_path)
            self.detect_img(image_or)

    def detect_img(self, image_or):
        image, ratio, dwdh = self.letterbox(image_or, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32) / 255.0

        session = self.ort_session
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]: im}

        outputs = session.run(outname, inp)[0]
        output = torch.from_numpy(outputs)
        out = self.non_max_suppression_face(output, self.conf_thres, self.iou_thres)[0]

        img = self.result(image_or, ratio, dwdh, out)
        if self.webcam:
            return img
        else:
            cv2.imwrite('./result.jpg', img)

    def result(self, img, ratio, dwdh, out):
        names = self.class_name()
        colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}

        for i, (x0, y0, x1, y1, score) in enumerate(out[:, 0:5]):
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            score = round(float(score), 3)
            name_label = names[0]
            color = colors[name_label]

            cropped_box = img[box[1]:box[3], box[0]:box[2]]
            if cropped_box.shape[0] == 0 or cropped_box.shape[1] == 0:
                continue

            cropped_box = cv2.resize(cropped_box, (112, 112))
            cropped_box = cv2.cvtColor(cropped_box, cv2.COLOR_BGR2RGB)
            cropped_box = np.transpose(cropped_box, (2, 0, 1))
            cropped_box = torch.from_numpy(cropped_box).unsqueeze(0).float().to(self.device)
            cropped_box.div_(255).sub_(0.5).div_(0.5)

            # 提取特征并归一化
            with torch.no_grad():
                feat = self.net(cropped_box).cpu().numpy()
            feat = feat / np.linalg.norm(feat)

            # 与特征库比对（欧氏距离）
            ts = os.listdir('database_tensor')
            if ts:
                best_name = "unknown"
                best_dist = float('inf')
                for tname in ts:
                    if not tname.endswith('.npy'):
                        continue
                    tmp = np.load(os.path.join('database_tensor', tname))
                    dist = np.linalg.norm(feat - tmp)
                    if dist < best_dist:
                        best_dist = dist
                        best_name = tname.replace('.npy', '')
                name_label = best_name if best_dist < 1.5 else "unknown"

            cv2.rectangle(img, box[:2], box[2:], color, 2)
            cv2.putText(img, f"{name_label} {score:.2f}", (box[0], box[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
        return img

    def non_max_suppression_face(self, prediction, conf_thres=0.25, iou_thres=0.45,
                                  classes=None, agnostic=False, labels=()):
        nc = prediction.shape[2] - 15
        xc = prediction[..., 4] > conf_thres
        max_wh = 4096
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

    @staticmethod
    def box_iou(box1, box2):
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + 1e-7)

    @staticmethod
    def xywh2xyxy(x):
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def class_name(self):
        classes = []
        with open(self.names, 'r') as file:
            while True:
                name = file.readline().strip('\n')
                classes.append(name)
                if not name:
                    break
        return classes

    @staticmethod
    def letterbox(im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        shape = im.shape[:2]
        new_shape = (640, 640)  # default
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
    image = './/yolov5-face//data/images/test.jpg'
    weights = './yolov5m-face.onnx'
    conf = 0.7
    iou_thres = 0.5
    img_size = 640
    classes_txt = './/yolov5-face//classes.txt'

    ORT = ort_v5(image, weights, conf, iou_thres, (img_size, img_size),
                 classes=classes_txt, webcam=True)
    ORT()
