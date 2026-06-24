"""
FaceID 一键启动脚本
运行: python run_detection.py

选项:
  --image <path>    使用图片模式
  --no_webcam       不使用摄像头
  --conf 0.7        置信度阈值
"""
import os, sys

# 确保在当前目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 检查关键文件
checks = [
    ("yolov5m-face.onnx", "YOLO检测模型"),
    ("backbone.pth", "人脸识别模型权重"),
    ("yolov5-face/classes.txt", "类别文件"),
    ("database_tensor", "人脸特征库"),
]

print("=" * 50)
print("  FaceID 系统检查")
print("=" * 50)
all_ok = True
for path, name in checks:
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {name}: {path}")
    if not exists:
        all_ok = False

if not os.path.exists("database_tensor") or len(os.listdir("database_tensor")) == 0:
    print(f"  ⚠️  database_tensor 为空，请先运行 feature_extraction.py")
    print(f"     conda activate GJSDXX")
    print(f"     python feature_extraction.py --weight backbone.pth --path_database database_image")
print("=" * 50)

if not all_ok:
    print("[WARN] 部分文件缺失，某些功能可能不可用")
    proceed = input("是否继续启动？(Y/n): ").strip().lower()
    if proceed == "n":
        sys.exit(0)

# 从命令行参数读取
import argparse
parser = argparse.ArgumentParser(description="FaceID: YOLOv5 + ArcFace 人脸检测识别")
parser.add_argument("--image", type=str, default=None, help="图片路径")
parser.add_argument("--no_webcam", action="store_true", help="不使用摄像头")
parser.add_argument("--conf", type=float, default=0.7, help="置信度阈值")
parser.add_argument("--iou", type=float, default=0.5, help="IOU阈值")
parser.add_argument("--yolo", type=str, default="yolov5m-face.onnx", help="YOLO ONNX模型")
parser.add_argument("--backbone", type=str, default="backbone.pth", help="ArcFace权重")
parser.add_argument("--classes", type=str, default=".\\yolov5-face\\classes.txt", help="类别文件")
parser.add_argument("--img_size", type=int, default=640, help="输入大小")
args = parser.parse_args()

webcam_mode = not args.no_webcam
img_path = args.image or ".\\yolov5-face\\data\\images\\test.jpg"

# 导入并运行
from detection_gpu import FaceDetector

print(f"\n[启动] 模式: {'摄像头' if webcam_mode else '图片'}")
print(f"[启动] 按 Q 退出, 按 S 截图\n")

det = FaceDetector(
    img_path,
    yolo_model=args.yolo,
    backbone_model=args.backbone,
    conf_thres=args.conf,
    iou_thres=args.iou,
    img_size=(args.img_size, args.img_size),
    classes=args.classes,
    webcam=webcam_mode
)
det()
