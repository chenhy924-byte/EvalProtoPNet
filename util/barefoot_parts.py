import os
from functools import lru_cache


def in_bbox(loc, bbox):
    return loc[0] >= bbox[0] and loc[0] <= bbox[1] and loc[1] >= bbox[2] and loc[1] <= bbox[3]


@lru_cache(maxsize=8)
def load_barefoot_parts(data_root: str):
    """
    Load barefoot keypoint annotations in a CUB-like format from `data_root`, returning:
      id_to_path: {img_id: (class_folder, filename)}
      id_to_part_loc: {img_id: [[part_id, x, y], ...]}  (only visible keypoints)
      id_to_bbox: {img_id: (x1, y1, x2, y2)} (full-image bbox as images are already cropped)
      part_num: int

    Expected files under data_root:
      images.txt
      image_class_labels.txt
      train_test_split.txt
      parts/parts.txt
      parts/part_locs.txt
    """
    img_txt = os.path.join(data_root, 'images.txt')
    part_cls_txt = os.path.join(data_root, 'parts', 'parts.txt')
    part_loc_txt = os.path.join(data_root, 'parts', 'part_locs.txt')

    id_to_path = {}
    with open(img_txt, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            img_id, img_path = int(line.split(' ')[0]), line.split(' ')[1].strip()
            # examples: test_cropped/000/xxx.jpg
            parts = img_path.split('/')
            if len(parts) >= 3:
                id_to_path[img_id] = (parts[-2], parts[-1])

    part_id_to_part = {}
    with open(part_cls_txt, 'r', encoding='utf-8') as f:
        for part_cls_line in f:
            if not part_cls_line.strip():
                continue
            id_len = len(part_cls_line.split(' ')[0])
            part_id, part_name = part_cls_line[:id_len], part_cls_line[id_len + 1:].strip()
            part_id_to_part[part_id] = part_name
    part_num = len(part_id_to_part.keys())

    id_to_part_loc = {}
    with open(part_loc_txt, 'r', encoding='utf-8') as f:
        for part_loc_line in f:
            if not part_loc_line.strip():
                continue
            content = part_loc_line.split(' ')
            img_id = int(content[0])
            part_id = int(content[1])
            loc_x = int(float(content[2]))
            loc_y = int(float(content[3]))
            visible = int(content[4])
            if visible != 1:
                continue
            id_to_part_loc.setdefault(img_id, []).append([part_id, loc_x, loc_y])

    # Barefoot images are already cropped; treat full image as bbox.
    # Keypoints are labeled on the cropped images; 224x224 is the default training size.
    default_bbox = (0, 0, 224, 224)
    id_to_bbox = {img_id: default_bbox for img_id in id_to_path.keys()}

    return id_to_path, id_to_part_loc, id_to_bbox, part_num


