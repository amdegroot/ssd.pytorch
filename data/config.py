# config.py
# 配置文件包含数据集信息的配置和网络结构的配置（lr, decay_step, anchor的配置等）
# 即配置文件说明使用哪一个数据集，使用哪一个版本的ssd
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc300 = {
    'num_classes': 21,  # 0-19 object, 20 background
    # 'lr_steps': (80000, 100000, 120000),
    # 'max_iter': 120000,
    'lr_steps': (8000, 10000, 12000),
    'max_iter': 12000,
    'means': (104, 117, 123),
    'feature_maps': [38, 19, 10, 5, 3, 1],  # modified
    # 分辨率
    'min_dim': 300, # modified
    'steps': [8, 16, 32, 64, 100, 300], # modified
    'min_sizes': [30, 60, 120, 162, 213, 264],  # modified
    'max_sizes': [60, 120, 162, 213, 264, 315], # modified
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# SSD512 CONFIGS
voc512 = {
    'num_classes': 21,  # 0-19 object, 20 background
    # 'lr_steps': (80000, 100000, 120000),
    # 'max_iter': 120000,
    'lr_steps': (8000, 10000, 12000),
    'max_iter': 12000,
    'means': (104, 117, 123),
    'feature_maps': [64, 32, 16, 8, 6, 4, 2],
    'min_dim': 512, # 分辨率
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [20, 51, 133, 215, 297, 379, 461],
    'max_sizes': [51, 133, 215, 297, 379, 461, 543],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# using ssd 512 configuration
# 表明使用voc0712做训练测试时使用SSD512的配置
voc=voc300

visdrone300 = {    # 意思是visdrone数据集跑ssd300
    'num_classes': 12,  # 包含ignored class
    'lr_steps': (8000, 10000, 12000),
    # 'max_iter': 12000,
    'max_iter': 12000,
    'means': (119, 122, 116),
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VisDrone2018',
}

visdrone512 = {
    'num_classes': 12,  # 包含ignored class
    'lr_steps': (80000, 100000, 120000),
    # 'max_iter': 12000,
    'max_iter': 120000,
    'means': (119, 122, 116),
    'feature_maps': [64, 32, 16, 8, 6, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [20, 51, 133, 215, 297, 379, 461],
    'max_sizes': [51, 133, 215, 297, 379, 461, 543],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VisDrone2018',
}

# Change SSD configuration
visdrone=visdrone512

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'means': (104, 117, 123),
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
