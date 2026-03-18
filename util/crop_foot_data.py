import argparse
import logging
import multiprocessing as mp
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────
#  全局配置参数
# ─────────────────────────────────────────────────────

OUTPUT_SIZE = 224  # 输出图像尺寸
BINARY_THRESH = 10  # 二值化阈值
DILATE_ITER = 2  # 膨胀迭代次数
MIN_CONTOUR_AREA = 4000  # 最小轮廓面积
MIN_WIDTH = 85  # 最小宽度
MIN_HEIGHT = 60  # 最小高度
MIN_TIGHT_W = 75  # 紧凑框最小宽度
MAX_TIGHT_W = 155  # 紧凑框最大宽度
TIGHT_H_THRESHOLD = 65  # 高度判定阈值
MIN_ACTUAL_H = 30  # 实际最小高度
TRUNC_RATIO = 0.20  # 截断比例阈值
EDGE_MARGIN = 50  # 边缘留白
SKEW_THRESHOLD = 8.0  # 倾斜矫正阈值
CROP_PAD = 3  # 裁剪填充
SPLIT_RATIO = 0.2  # 测试集比例 (20%)


# ─────────────────────────────────────────────────────
#  图像处理工具函数
# ─────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    """配置日志记录器"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("footprint_process")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(str(log_path), encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def pad_to_square(img_bgr: np.ndarray, size: int, scale: float) -> np.ndarray:
    """等比例缩放并填充为 size x size 的黑色背景正方形"""
    h, w = img_bgr.shape[:2]
    new_h = max(1, min(size, int(round(h * scale))))
    new_w = max(1, min(size, int(round(w * scale))))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0: y0 + new_h, x0: x0 + new_w] = resized
    return canvas


def correct_skew(crop_bgr: np.ndarray, binary_crop: np.ndarray) -> np.ndarray:
    """检测并校正足迹的倾斜角度"""
    pts = cv2.findNonZero(binary_crop)
    if pts is None or len(pts) < 10:
        return crop_bgr
    _, (rw, rh), angle = cv2.minAreaRect(pts)
    if rw < rh: angle += 90.0
    if abs(angle) <= SKEW_THRESHOLD:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin_a + w * cos_a), int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(crop_bgr, M, (new_w, new_h), borderValue=(0, 0, 0))


def _is_complete_foot(binary_rotated: np.ndarray) -> bool:
    """通过检测前景块的垂直分布，判断是否同时包含脚掌和足跟"""
    if binary_rotated is None or not binary_rotated.any():
        return False
    h = binary_rotated.shape[0]
    row_active = binary_rotated.any(axis=1)
    segments = []
    in_seg, start = False, 0
    for y, active in enumerate(row_active):
        if active and not in_seg:
            in_seg, start = True, y
        elif not active and in_seg:
            in_seg = False
            segments.append((start, y))
    if in_seg: segments.append((start, h))

    main_segments = [(s, e) for (s, e) in segments if (e - s) >= max(4, int(0.08 * h))]
    if len(main_segments) < 2:
        return False
    top_center = (main_segments[0][0] + main_segments[0][1]) / 2.0
    bottom_center = (main_segments[-1][0] + main_segments[-1][1]) / 2.0
    return (bottom_center - top_center) >= (0.35 * h)


# ─────────────────────────────────────────────────────
#  足迹提取核心逻辑
# ─────────────────────────────────────────────────────

def try_extract(img: np.ndarray, binary: np.ndarray, kernel_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """使用指定核大小进行闭运算并提取足迹候选区域"""
    img_h, img_w = img.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=DILATE_ITER)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA or w < MIN_WIDTH or h < MIN_HEIGHT:
            continue
        if (x < EDGE_MARGIN or x + w > img_w - EDGE_MARGIN) and w < 140:
            continue

        roi_bin = binary[y: y + h, x: x + w]
        pts = cv2.findNonZero(roi_bin)
        if pts is None: continue
        tx, ty, tw, th = cv2.boundingRect(pts)
        if not (MIN_TIGHT_W <= tw <= MAX_TIGHT_W): continue

        fx, fy = max(0, x + tx - CROP_PAD), max(0, y + ty - CROP_PAD)
        fw, fh = min(img_w - fx, tw + CROP_PAD * 2), min(img_h - fy, th + CROP_PAD * 2)

        crop = img[fy: fy + fh, fx: fx + fw]
        crop_bin = binary[fy: fy + fh, fx: fx + fw]
        if crop.size > 0:
            results.append((crop, crop_bin))
    return results


def process_one_folder(task: Dict) -> Dict:
    """多进程任务单元：处理单个编号文件夹"""
    num_str = task["num_str"]
    bmp_path = Path(task["bmp_path"])
    dst_jpg_dir = Path(task["dst_jpg_dir"])
    res = {"num_str": num_str, "saved": 0, "errors": []}

    try:
        raw = np.fromfile(str(bmp_path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None: raise ValueError(f"解码失败: {bmp_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, BINARY_THRESH, 255, cv2.THRESH_BINARY)

        # 提取并过滤完整足迹
        candidates = try_extract(img, binary, 25)
        footprints = []
        for crop, crop_bin in candidates:
            corrected = correct_skew(crop, crop_bin)
            rot_bgr = cv2.rotate(corrected, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rot_bin = cv2.rotate(crop_bin, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if _is_complete_foot(rot_bin):
                footprints.append(rot_bgr)

        if not footprints:
            res["errors"].append("未发现完整足迹")
            return res

        # 计算统一缩放比
        median_max = np.median([max(f.shape[0], f.shape[1]) for f in footprints])
        scale = OUTPUT_SIZE / median_max if median_max > 0 else 0

        if scale <= 0: return res

        dst_jpg_dir.mkdir(parents=True, exist_ok=True)
        for idx, fp in enumerate(footprints):
            padded = pad_to_square(fp, OUTPUT_SIZE, scale)
            out_name = f"{num_str}_{idx}_{task['person_tail']}.jpg"
            ok, buf = cv2.imencode(".jpg", padded, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                buf.tofile(str(dst_jpg_dir / out_name))
                res["saved"] += 1

    except Exception as e:
        res["errors"].append(str(e))
    return res


# ─────────────────────────────────────────────────────
#  数据集划分逻辑
# ─────────────────────────────────────────────────────

def split_dataset(jpg_root: Path, split_root: Path, logger: logging.Logger):
    """将提取好的 JPG 按人员编号划分为训练集和测试集"""
    train_root = split_root / "train_croped"
    test_root = split_root / "test_croped"

    subfolders = [f for f in jpg_root.iterdir() if f.is_dir()]
    total_train, total_test = 0, 0

    logger.info("开始划分数据集...")
    for folder in subfolders:
        images = [img for img in folder.iterdir() if img.suffix.lower() in ('.jpg', '.jpeg')]
        if not images: continue

        random.shuffle(images)
        test_count = max(1, int(len(images) * SPLIT_RATIO))
        if len(images) > 1 and test_count >= len(images):
            test_count = len(images) - 1

        test_images = images[:test_count]
        train_images = images[test_count:]

        for img_list, target_root in [(train_images, train_root), (test_images, test_root)]:
            dest_dir = target_root / folder.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in img_list:
                shutil.copy(img, dest_dir / img.name)

        total_train += len(train_images)
        total_test += len(test_images)

    logger.info(f"划分完成！训练集: {total_train} 张, 测试集: {total_test} 张")
    logger.info(f"结果保存在: {split_root}")


# ─────────────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="足迹自动提取与数据集划分整合脚本")
    parser.add_argument("--input-root", type=str, required=True, help="输入根目录 (000~199 子目录)")
    parser.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() - 1), help="进程数")
    args = parser.parse_args()

    desktop = Path.home() / "Desktop"
    src_root = Path(args.input_root).expanduser().resolve()
    jpg_root = desktop / "Barefoot_Dataset_undivided"
    split_root = desktop / "Barefoot_Dataset"
    log_path = desktop / "footprint_process_log.txt"

    logger = setup_logger(log_path)
    logger.info("=" * 50)
    logger.info("第一阶段：足迹提取启动")

    # 收集任务
    tasks = []
    folders = sorted([p for p in src_root.iterdir() if p.is_dir() and p.name.isdigit()],
                     key=lambda p: int(p.name))

    for folder in folders:
        bmp_files = list(folder.glob("*.bmp"))
        for bmp in bmp_files:
            tail = bmp.stem[len(folder.name):].lstrip("_")
            tasks.append({
                "num_str": folder.name,
                "bmp_path": str(bmp),
                "person_tail": tail,
                "dst_jpg_dir": str(jpg_root / folder.name)
            })

    # 多进程提取
    total_saved = 0
    with mp.Pool(processes=args.num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_one_folder, tasks), total=len(tasks), desc="提取进度"):
            total_saved += res["saved"]
            for msg in res["errors"]:
                logger.debug(f"[{res['num_str']}] {msg}")

    logger.info(f"提取阶段结束，成功保存 {total_saved} 张 JPG")

    # 第二阶段：划分
    if total_saved > 0:
        logger.info("=" * 50)
        logger.info("第二阶段：数据集自动划分启动")
        split_dataset(jpg_root, split_root, logger)
    else:
        logger.error("未发现可划分的足迹图像。")


if __name__ == "__main__":
    mp.freeze_support()
    main()

# python crop_foot_data.py --input-root "C:\Users\31877\Desktop\yali"
