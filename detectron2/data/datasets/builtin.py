# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc
from .kitti import register_kitti

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    # "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    # "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    # "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    # "coco_2014_valminusminival": (
    #     "coco/val2014",
    #     "coco/annotations/instances_valminusminival2014.json",
    # ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_tta_train": ("coco/val2017", "coco/annotations/instances_tta_train2017.json"),
    "coco_2017_tta_val": ("coco/val2017", "coco/annotations/instances_tta_val2017.json"),
    # "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    # "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    # "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    # "keypoints_coco_2014_train": (
    #     "coco/train2014",
    #     "coco/annotations/person_keypoints_train2014.json",
    # ),
    # "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    # "keypoints_coco_2014_minival": (
    #     "coco/val2014",
    #     "coco/annotations/person_keypoints_minival2014.json",
    # ),
    # "keypoints_coco_2014_valminusminival": (
    #     "coco/val2014",
    #     "coco/annotations/person_keypoints_valminusminival2014.json",
    # ),
    # "keypoints_coco_2017_train": (
    #     "coco/train2017",
    #     "coco/annotations/person_keypoints_train2017.json",
    # ),
    # "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    # "keypoints_coco_2017_val_100": (
    #     "coco/val2017",
    #     "coco/annotations/person_keypoints_val2017_100.json",
    # ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    # "coco_2017_train_panoptic": (
    #     # This is the original panoptic annotation directory
    #     "coco/panoptic_train2017",
    #     "coco/annotations/panoptic_train2017.json",
    #     # This directory contains semantic annotations that are
    #     # converted from panoptic annotations.
    #     # It is used by PanopticFPN.
    #     # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
    #     # to create these directories.
    #     "coco/panoptic_stuff_train2017",
    # ),
    # "coco_2017_val_panoptic": (
    #     "coco/panoptic_val2017",
    #     "coco/annotations/panoptic_val2017.json",
    #     "coco/panoptic_stuff_val2017",
    # ),
    # "coco_2017_val_100_panoptic": (
    #     "coco/panoptic_val2017_100",
    #     "coco/annotations/panoptic_val2017_100.json",
    #     "coco/panoptic_stuff_val2017_100",
    # ),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
            from imagecorruptions import get_corruption_names
            for corrupt in get_corruption_names():
                register_coco_instances(
                    "{}-{}".format(key, corrupt),
                    _get_builtin_metadata("coco"),
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                    corrupt=corrupt,
                )
        #register_coco_instances(
        #    "coco_2017_train-{}".format(corrupt),
        #    _get_builtin_metadata("coco"),
        #    os.path.join(root, json_file) if "://" not in json_file else json_file,
        #    os.path.join(root, image_root),
        #    corrupt=corrupt,
        #)
        #import pdb; pdb.set_trace()
        #register_coco_instances(
        #    "coco_2017_tta_val-{}".format(corrupt),
        #    _get_builtin_metadata("coco"),
        #    os.path.join(root, json_file) if "://" not in json_file else json_file,
        #    os.path.join(root, image_root),
        #    corrupt=corrupt,
        #)
        #register_coco_instances(
        #    "coco_2017_tta_train-{}".format(corrupt),
        #    _get_builtin_metadata("coco"),
        #    os.path.join(root, json_file) if "://" not in json_file else json_file,
        #    os.path.join(root, image_root),
        #    corrupt=corrupt,
        #)
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

    from imagecorruptions import get_corruption_names
    for corrupt in get_corruption_names():
        register_pascal_voc("voc_2007_test-{}".format(corrupt), os.path.join(root, "VOC2007"), "test", "2007", corrupt=corrupt)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

def register_all_kitti(root):
    SPLITS = [
        ("kitti_train", "KITTI", "train"),
        ("kitti_val", "KITTI", "val"),
    ]
    for name, dirname, split in SPLITS:
        register_kitti(name, os.path.join(root, dirname), split)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        if split == 'val':
            for corrupt in ['fog', 'rain', 'snow']:
                register_kitti('{}-{}'.format(name, corrupt), os.path.join('./datasets', dirname), split, corrupt=corrupt)
                MetadataCatalog.get('{}-{}'.format(name, corrupt)).evaluator_type = "pascal_voc"

def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

################# USER defined ##############
def load_shift_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, corrupt=None):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with open(json_file, 'r') as f:
        anno = json.load(f)
    
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
   
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0
    _image_root = image_root if corrupt is None else "{}-{}".format(image_root, corrupt)
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(_image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts

def register_all_shift(root):
    from .coco import load_coco_json
    SHIFT_CATEGORIES = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]
    _SHIFT_SPLITS = {
        "shift_{division}_train": ("shift/{division}/images/train/front/images", "shift/{division}/images/train/front/"),
        "shift_{division}_val": ("shift/{division}/images/val/front/images", "shift/{division}/images/val/front/"),
    }
    DIVISION = ['discrete', 'continuous1x', 'continuous10x']
    
    for div in DIVISION:
        for _name, (_img_dir, _gt_dir) in _SHIFT_SPLITS.items():
            name, img_dir, gt_dir = _name.format(division=div), \
                os.path.join(root, _img_dir.format(division=div)), os.path.join(root, _gt_dir.format(division=div))
            DatasetCatalog.register(name, lambda: load_coco_json(os.path.join(gt_dir, 'det_2d.json'), img_dir, dataset_name=None))
            MetadataCatalog.get(name).set(
                stuff_classes=SHIFT_CATEGORIES,
                image_root=img_dir,
                evaluator_type="coco",
                gt_dir=gt_dir,
            )
            
##############################################

# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)
    register_all_kitti(_root)
