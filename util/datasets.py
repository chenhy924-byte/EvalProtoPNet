import os
import logging
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Cub2011Eval(Dataset):
    base_folder = 'test_cropped'

    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        img_id = sample.img_id

        if self.transform is not None:
            img = self.transform(img)

        return img, target, img_id


class Barefoot_Dataset(Dataset):
    """
    赤足压力足迹数据集：从 train_cropped_augmented / test_cropped 加载 JPG 压力图。
    结构与 ImageFolder 一致（每类一个子文件夹，类名为文件夹名如 000, 001, ...）。
    损坏图片在初始化时检测并跳过，并记录日志。
    """
    TRAIN_DIR = 'train_cropped_augmented'
    TEST_DIR = 'test_cropped'

    def __init__(self, root, train=True, transform=None, check_integrity=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        if train:
            subdir = self.TRAIN_DIR
            if not os.path.isdir(os.path.join(self.root, subdir)):
                subdir = 'train_cropped'
            self.data_dir = os.path.join(self.root, subdir)
        else:
            self.data_dir = os.path.join(self.root, self.TEST_DIR)

        if not os.path.isdir(self.data_dir):
            raise RuntimeError('Barefoot dataset not found: {}'.format(self.data_dir))

        # 类别文件夹排序，保证类别索引稳定
        self.class_names = sorted([d for d in os.listdir(self.data_dir)
                                  if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # 收集 (path, class_idx)，可选完整性检查并跳过损坏图片
        self.samples = []
        self.skipped = []
        all_candidates = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    path = os.path.join(class_dir, fname)
                    all_candidates.append((path, self.class_to_idx[class_name]))

        if check_integrity and all_candidates:
            for path, class_idx in tqdm(all_candidates, desc='Checking images', unit='img'):
                try:
                    img = self.loader(path)
                    if img is not None:
                        self.samples.append((path, class_idx))
                except Exception as e:
                    self.skipped.append((path, str(e)))
                    logger.warning('Skip corrupted image: %s - %s', path, e)
        else:
            self.samples = list(all_candidates)

        if self.skipped:
            logger.info('Barefoot_Dataset: skipped %d corrupted images (see warnings above).', len(self.skipped))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            img = self.loader(path)
        except Exception as e:
            logger.warning('Failed to load image at runtime: %s - %s', path, e)
            # 若运行时仍失败，用同类别第一张图替代，避免断训
            for i, (p, t) in enumerate(self.samples):
                if t == target and p != path:
                    path = p
                    img = self.loader(path)
                    break
            else:
                raise RuntimeError('Cannot recover from corrupted image: {}'.format(path))
        if self.transform is not None:
            img = self.transform(img)
        return img, target