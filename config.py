# TODO: automatic detect directory
code_dir = "/Users/xiaohan/research/Equisetum_new"

cfg = {
    # dataset name
    'name': 'Equisetum',

    # file directory
    'img_dir': code_dir + '/Equisetum_dataset/Images',
    'anno_dir': code_dir + '/Equisetum_dataset/Annotations',
    # 'training_set': '/Users/xiaohan/research/Equisetum/code/data_processing/HL_training_set.json',
    # 'val_set': '/Users/xiaohan/research/Equisetum/code/data_processing/HL_val_set.json',
    # 'test_set': '/Users/xiaohan/research/Equisetum/code/data_processing/HL_test_set.json',
    # 'save_folder': '/Users/xiaohan/research/Equisetum/code/weights/',

    # # remote direction
    # 'img_dir': '/home/home1/xw176/work/Equisetum/dataset/img',
    # 'anno_dir': '/home/home1/xw176/work/Equisetum/dataset/anno',
    # 'training_set': '/home/home1/xw176/work/Equisetum/data_processing/HL_training_set.json',
    # 'val_set': '/home/home1/xw176/work/Equisetum/data_processing/HL_val_set.json',
    # 'test_set': '/home/home1/xw176/work/Equisetum/data_processing/HL_test_set.json',
    # 'save_folder': '/home/home1/xw176/work/Equisetum/weights/',

    'class': {
        'hyemale': 0,
        'laevigatum': 1,
        'ferrissii': 2
    },

    # 'HL_map_anno': {
    #     0: 'bg',
    #     1: 'HNode',
    #     2: 'LNode'
    # },

    'is_useful_anno': {
        'SteNodeNor': 1,
        'Strobilus': 0,
        'SteInter': 0,
        'SteNodeInj': 1,
        'Undefined': 0
    },

    # # 'train_num': 90,
    #
    # 'mean': [218, 213, 194],
    #
    # # ssd cofig
    # 'num_classes': 3,
    # 'min_dim': 300,
    #
    # # training process config
    # 'num_workers': 0,
    # 'batch_size': 8,
    # 'max_iter': 12000,
    # 'lr': 1e-4,
    # 'lr_steps': (80000, 100000, 120000),
    # 'feature_maps': [38, 19, 10, 5, 3, 1],
    # 'min_dim': 300,
    # 'steps': [8, 16, 32, 64, 100, 300],
    # 'min_sizes': [30, 60, 111, 162, 213, 264],
    # 'max_sizes': [60, 111, 162, 213, 264, 315],
    # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    # 'variance': [0.1, 0.2],
    # 'clip': True,
    # 'eval_folder': 'eval/'
}
