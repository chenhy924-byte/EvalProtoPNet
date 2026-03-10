import Augmentor
import os
import argparse

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='datasets/Barefoot_Dataset/',
                    help='数据集根目录，内含 train_cropped/；默认: datasets/Barefoot_Dataset/')
args = parser.parse_args()

datasets_root_dir = args.data_path
dir = os.path.join(datasets_root_dir, 'train_cropped')
target_dir = os.path.join(datasets_root_dir, 'train_cropped_augmented')
# 使用绝对路径，避免 Augmentor 将 output_directory 解析到 source 下
root_abs = os.path.abspath(os.path.normpath(datasets_root_dir))
dir_abs = os.path.abspath(dir)
target_dir_abs = os.path.abspath(target_dir)

makedir(target_dir_abs)
folders = [os.path.join(dir_abs, folder) for folder in next(os.walk(dir_abs))[1]]
target_folders = [os.path.join(target_dir_abs, folder) for folder in next(os.walk(dir_abs))[1]]

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]
    # 跳过空类别文件夹，避免 Augmentor "no images in the pipeline" 报错
    num_imgs = len([f for f in os.listdir(fd) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    if num_imgs == 0:
        continue
    # rotation
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    for _ in range(10):
        p.process()
    del p
    # skew
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for _ in range(10):
        p.process()
    del p
    # shear
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)
    for _ in range(10):
        p.process()
    del p
    # random_distortion
    #p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    #p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    #p.flip_left_right(probability=0.5)
    #for i in range(10):
    #    p.process()
    #del p