import argparse
import os
import cv2
import numpy as np
from pathlib import Path


def extract_feature_onnx(session, img_path_or_array):
    """用ONNX模型提取人脸特征 (512维)"""
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path_or_array}")
    else:
        img = img_path_or_array

    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32)
    img = img / 255.0
    img = (img - 0.5) / 0.5

    feat = session.run(['output'], {'input': img})[0]
    return feat


def extract_feature_torch(net, device, img_path_or_array):
    """用PyTorch模型提取人脸特征 (512维)，自动GPU加速"""
    import torch
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path_or_array}")
    else:
        img = img_path_or_array

    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    img.div_(255).sub_(0.5).div_(0.5)

    with torch.no_grad():
        feat = net(img).cpu().numpy()
    return feat


def export_to_onnx(weight_path, output_path='backbone.onnx'):
    """将PyTorch权重导出为ONNX格式"""
    import torch
    from iresnet import iresnet100

    print(f"[INFO] 加载 PyTorch 权重: {weight_path}")
    net = iresnet100(False)
    net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    net.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    torch.onnx.export(
        net, dummy_input, output_path,
        input_names=['input'], output_names=['output'],
        opset_version=18
    )
    print(f"[OK] ONNX模型已导出: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceID 特征提取工具")
    parser.add_argument("--weight", type=str, default=None,
                        help="PyTorch权重文件路径 (.pth)，不提供则使用 backbone.onnx")
    parser.add_argument("--onnx", type=str, default="backbone.onnx",
                        help="ONNX模型路径 (默认: backbone.onnx)")
    parser.add_argument("--path_database", type=Path, default=None,
                        help="人脸图片目录，提取特征保存到 database_tensor/")
    parser.add_argument("--export_onnx", action="store_true",
                        help="将PyTorch权重导出为ONNX")
    parser.add_argument("--single", type=str, default=None,
                        help="单张图片路径，提取特征并保存")
    args = parser.parse_args()

    os.makedirs("database_tensor", exist_ok=True)

    # ---- 先处理导出ONNX ----
    if args.export_onnx:
        if not args.weight:
            print("[ERROR] --export_onnx 需要 --weight 参数")
            exit(1)
        export_to_onnx(args.weight, args.onnx)

    # ---- 特征提取 ----
    session = None
    net = None
    device = None
    use_torch = False

    if args.weight:
        # PyTorch 模式
        import torch
        from iresnet import iresnet100
        print(f"[INFO] 加载 PyTorch 权重: {args.weight}")
        net = iresnet100(False)
        net.load_state_dict(torch.load(args.weight, map_location='cpu'))
        net.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        use_torch = True
        print(f"[INFO] 使用设备: {device}")
    else:
        # ONNX 模式
        import onnxruntime as ort
        if not os.path.exists(args.onnx):
            print(f"[ERROR] ONNX模型不存在: {args.onnx}")
            print("  请先提供 --weight 参数加载PyTorch权重，或确保 backbone.onnx 存在")
            exit(1)
        print(f"[INFO] 加载 ONNX 模型: {args.onnx}")
        session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    # ---- 处理图片 ----
    img_list = []

    if args.single:
        img_list.append(args.single)

    if args.path_database:
        path = args.path_database
        imgs = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for im in imgs:
            img_list.append(os.path.join(path, im))

    if not img_list:
        print("[INFO] 未指定图片路径，仅完成模型加载")
        exit(0)

    for img_path in img_list:
        if use_torch:
            feat = extract_feature_torch(net, device, img_path)
        else:
            feat = extract_feature_onnx(session, img_path)

        # L2 归一化后保存
        feat = feat / np.linalg.norm(feat)

        save_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.npy'
        save_path = os.path.join('database_tensor', save_name)
        np.save(save_path, feat)
        print(f"  [OK] {os.path.basename(img_path)} -> {save_path} (归一化)")

    print(f"[完成] 共处理 {len(img_list)} 张图片")
