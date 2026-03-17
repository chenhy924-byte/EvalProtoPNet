import os
import json
import logging
from tqdm import tqdm

# === 1. 配置路径与日志 ===
json_root = r'D:\SeniorThesis\EvalProtoPNet\datasets\yali_200_jpg_label'
img_dataset_root = r'D:\SeniorThesis\EvalProtoPNet\datasets\Barefoot_Dataset'
parts_out_dir = os.path.join(img_dataset_root, 'parts')
log_file = os.path.join(img_dataset_root, 'conversion_log.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8')]
)

if not os.path.exists(parts_out_dir):
    os.makedirs(parts_out_dir)

part_names = {str(i): f"point_{i}" for i in range(1, 10)}
sub_folders = {'test_cropped': 0, 'train_cropped': 1, 'train_cropped_augmented': 1}

# === 2. 阶段一：快速扫描结构 (确定总任务量) ===
print(">>> 阶段 1/2: 正在扫描数据集结构...")
all_tasks = []
classes_dict = {}

for folder_name, is_train in sub_folders.items():
    folder_path = os.path.join(img_dataset_root, folder_name)
    if not os.path.exists(folder_path): continue
    
    class_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    for class_name in class_folders:
        if class_name not in classes_dict:
            classes_dict[class_name] = len(classes_dict) + 1
        
        class_path = os.path.join(folder_path, class_name)
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for img in img_files:
            all_tasks.append({
                'folder': folder_name,
                'class': class_name,
                'img_name': img,
                'is_train': is_train,
                'class_id': classes_dict[class_name]
            })

# === 3. 阶段二：深度处理与 JSON 匹配 ===
print(f">>> 阶段 2/2: 正在匹配标注点并生成索引文件 (共 {len(all_tasks)} 张图像)...")
images_txt, image_class_labels, train_test_split, part_locs = [], [], [], []

# 使用一个全局进度条，并通过 set_description 显示实时详情
pbar = tqdm(total=len(all_tasks), unit="img")
image_id = 1
json_hit_count = 0

for task in all_tasks:
    base_name = os.path.splitext(task['img_name'])[0]
    # 更新进度条左侧的文字提示
    pbar.set_description(f"[{task['folder']}] 类别 {task['class']}")
    
    # 记录基础信息
    rel_path = f"{task['folder']}/{task['class']}/{task['img_name']}"
    images_txt.append(f"{image_id} {rel_path}")
    image_class_labels.append(f"{image_id} {task['class_id']}")
    train_test_split.append(f"{image_id} {task['is_train']}")

    # 寻找 JSON
    json_found = False
    for root, _, files in os.walk(json_root):
        target_json = base_name + ".json"
        if target_json in files:
            try:
                with open(os.path.join(root, target_json), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    shapes = {s['label']: s['points'][0] for s in data.get('shapes', [])}
                    for p_id in range(1, 10):
                        p_str = str(p_id)
                        if p_str in shapes:
                            x, y = shapes[p_str]
                            part_locs.append(f"{image_id} {p_id} {x} {y} 1")
                        else:
                            part_locs.append(f"{image_id} {p_id} 0.0 0.0 0")
                json_found = True
                json_hit_count += 1
            except Exception as e:
                logging.error(f"解析失败: {target_json} | {e}")
            break
    
    if not json_found:
        for p_id in range(1, 10):
            part_locs.append(f"{image_id} {p_id} 0.0 0.0 0")

    # 更新进度条右侧的统计信息
    pbar.set_postfix({"已匹配JSON": json_hit_count})
    pbar.update(1)
    image_id += 1

pbar.close()

# === 4. 保存文件 ===
print(">>> 正在写入 .txt 索引文件...")
def write_txt(name, data_list):
    with open(os.path.join(img_dataset_root, name), 'w', encoding='utf-8') as f:
        f.write("\n".join(data_list) + "\n")

write_txt('images.txt', images_txt)
write_txt('image_class_labels.txt', image_class_labels)
write_txt('train_test_split.txt', train_test_split)
write_txt('classes.txt', [f"{v} {k}" for k, v in sorted(classes_dict.items(), key=lambda x: x[1])])

with open(os.path.join(parts_out_dir, 'part_locs.txt'), 'w', encoding='utf-8') as f:
    f.write("\n".join(part_locs) + "\n")
with open(os.path.join(parts_out_dir, 'parts.txt'), 'w', encoding='utf-8') as f:
    f.write("\n".join([f"{i} point_{i}" for i in range(1, 10)]) + "\n")

print(f"\n任务完成！总图像: {len(all_tasks)}, 成功关联标注: {json_hit_count}")