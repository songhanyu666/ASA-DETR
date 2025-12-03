"""
ASA-DETR推理脚本
"""
import argparse
import os
import cv2
import torch
from pathlib import Path

# 导入模型（需要根据实际实现调整）
# from models.asa_detr import ASADETR
# from utils.general import non_max_suppression, scale_coords


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ASA-DETR推理脚本')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--source', type=str, required=True,
                        help='输入图像/文件夹/视频路径')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='NMS IOU阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU设备ID')
    parser.add_argument('--save-dir', type=str, default='runs/detect',
                        help='结果保存路径')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存结果为txt文件')
    parser.add_argument('--save-conf', action='store_true',
                        help='在txt文件中保存置信度')
    parser.add_argument('--view-img', action='store_true',
                        help='显示结果')
    return parser.parse_args()


def load_model(weights, device):
    """加载模型"""
    # TODO: 实现模型加载
    # model = ASADETR()
    # checkpoint = torch.load(weights, map_location=device)
    # model.load_state_dict(checkpoint['model'])
    # model = model.to(device)
    # model.eval()
    # return model
    print(f'加载模型权重: {weights}')
    return None


def detect_image(model, img_path, args):
    """检测单张图像"""
    # TODO: 实现图像检测
    # 1. 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'无法读取图像: {img_path}')
        return
    
    # 2. 预处理
    # img_tensor = preprocess(img, args.img_size)
    
    # 3. 推理
    # with torch.no_grad():
    #     pred = model(img_tensor)
    
    # 4. 后处理
    # pred = non_max_suppression(pred, args.conf_thres, args.iou_thres)
    
    # 5. 绘制结果
    # for det in pred:
    #     if len(det):
    #         det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
    #         for *xyxy, conf, cls in det:
    #             cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), 
    #                          (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    #             cv2.putText(img, f'{conf:.2f}', (int(xyxy[0]), int(xyxy[1])-10),
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 6. 保存结果
    save_path = Path(args.save_dir) / Path(img_path).name
    cv2.imwrite(str(save_path), img)
    print(f'检测结果已保存: {save_path}')
    
    # 7. 显示结果
    if args.view_img:
        cv2.imshow('Detection', img)
        cv2.waitKey(0)


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = load_model(args.weights, device)
    
    # 获取输入源
    source = Path(args.source)
    
    if source.is_file():
        # 单个文件
        if source.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            detect_image(model, source, args)
        else:
            print(f'不支持的文件格式: {source.suffix}')
    
    elif source.is_dir():
        # 文件夹
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_files.extend(source.glob(ext))
        
        print(f'找到 {len(img_files)} 张图像')
        for img_file in img_files:
            detect_image(model, img_file, args)
    
    else:
        print(f'无效的输入源: {source}')
    
    print('推理完成！')
    print(f'结果保存在: {save_dir}')


if __name__ == '__main__':
    main()