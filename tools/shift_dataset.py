import os
import logging
import csv
import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from detectron2.structures import Boxes, BoxMode

logger = logging.getLogger(__name__)


def load_shift_json(json_file, image_root, single_folder=None, filter_weather='clear', filter_time='daytime'):
    import json
    timer = Timer()
    with open(json_file, 'r') as f:
        anno = json.load(f)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    frames = anno["frames"]
    config = anno["config"]
    category = config["categories"]
    cat2cat_id = {}
    for id, id_map in enumerate(category):
        cat2cat_id[id_map["name"]] = id
    
    logger.info("Loaded {} images in SHIFT format from {}".format(len(frames), json_file))

    dataset_dicts = []

    ######## SHIFT Dataset Annotation Format example #######
    # "frames" : [
    #   { "name": "00000000_img_front.jpg", 
    #   "videoName": "028b-3dfe", 
    #   "intrinsics": {"focal": [640.0, 640.0], "center": [640.0, 400.0], "skew": 0}, 
    #   "extrinsics": {"location": [20.097749710083008, 24.609487533569336, 1.540658950805664], "rotation": [0.10971069291456362, -0.14779845984413653, 0.15902175958366796]},
    #   "attributes": {"weather_coarse": "overcast", "timeofday_coarse": "daytime", ... , "shift_type": "daytime_to_night", "shift_length": "400"},
    #   "frameIndex": 0, 
    #   "labels": [{"id": "473", "attributes": {"type": "vehicle.tesla.cybertruck"}, "category": "truck", "box2d": {"x1": 753.0, "y1": 399.0, "x2": 788.0, "y2": 413.0}},
    #           ... ]
    #   }
    # ...
    # ],
    # "config" :
    # { "imageSize": {"width": 1280, "height": 800}, 
    #   "categories": [{"name": "pedestrian"}, {"name": "car"}, {"name": "truck"}, {"name": "bus"}, {"name": "motorcycle"}, {"name": "bicycle"}]
    # }
    
    if single_folder is not None:
        frames = [ frames[i] for i in range(len(frames)) if frames[i]["videoName"] == single_folder ]

    if filter_weather is not None:
        frames = [f for f in frames if f['attributes']['weather_coarse'] == filter_weather]
    if filter_time is not None:
        frames = [f for f in frames if f['attributes']['timeofday_coarse'] == filter_time]
    for idx, frame in enumerate(frames):
        record = {}
        img_file = record["file_name"] = os.path.join(image_root, frame["videoName"], frame["name"])
        record["image_id"] = idx
        record["height"] = config["imageSize"]["height"]
        record["width"] = config["imageSize"]["width"]
        
        objs = []
        obj_dict_list = frame["labels"]
        for obj in obj_dict_list:
            if "box2d" not in obj:
                raise ValueError(
                    f"Annotation of image {img_file} does not contains 'box2d' value! "
                )
            # obj["bbox"] = [ obj["box2d"][coor] for coor in ["x1", "y1", "x2", "y2"] ]
            # obj["bbox_mode"] = BoxMode.XYXY_ABS
            obj["bbox"] = [ obj["box2d"]["x1"], obj["box2d"]["y1"], obj["box2d"]["x2"] - obj["box2d"]["x1"], obj["box2d"]["y2"] - obj["box2d"]["y1"]]
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            annotation_category = obj["category"]
            try:
                obj["category_id"] = cat2cat_id[annotation_category]
            except KeyError as e:
                raise KeyError(
                    f"Encountered category={annotation_category} "
                    "but this id does not exist in 'categories_id' of the json file."
                ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


SHIFT_CATEGORIES = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]
_SHIFT_SPLITS = {
    "shift_{division}_train": ("shift/{division}/images/train/front/images", "shift/{division}/images/train/front/"),
    "shift_{division}_val": ("shift/{division}/images/val/front/images", "shift/{division}/images/val/front/"),
}
# DIVISION = ['discrete', 'continuous1x', 'continuous10x']
DIVISION = ['discrete']


def register_data(name, func, metadata):
    DatasetCatalog.register(name, func)
    MetadataCatalog.get(name).set(**metadata)

def register_discrete_shift(image_root, weather='clear', timeofday='daytime'):
    # discrete
    attr_name = ('-{}'.format(weather) if weather is not None else '') + ('-{}'.format(timeofday) if timeofday is not None else '')
    train_name = 'shift_discrete_train' + attr_name
    test_name = 'shift_discrete_val' + attr_name
    metadata = {'image_root': image_root, 'stuff_classes': SHIFT_CATEGORIES, 'thing_classes': SHIFT_CATEGORIES, 'evaluator_type': "coco"}
    #DatasetCatalog.register(train_name, lambda: load_shift_json(os.path.join(image_root, 'shift/discrete/images/train/front/det_2d.json'), os.path.join(image_root, 'shift/discrete/images/train/front/images'), filter_weather=weather, filter_time=timeofday))
    #MetadataCatalog.get(train_name).set(**metadata)
    DatasetCatalog.register(test_name, lambda: load_shift_json(os.path.join(image_root, 'shift/discrete/images/val/front/det_2d.json'), os.path.join(image_root, 'shift/discrete/images/val/front/images'), filter_weather=weather, filter_time=timeofday))
    MetadataCatalog.get(test_name).set(**metadata)
    # register_data(train_name, lambda: load_shift_json(os.path.join(image_root, 'shift/discrete/images/train/front/det_2d.json'), os.path.join(image_root, 'shift/discrete/images/train/front/images'), filter_weather=weather, filter_time=timeofday), metadata)
    # register_data(test_name, lambda: load_shift_json(os.path.join(image_root, 'shift/discrete/images/val/front/det_2d.json'), os.path.join(image_root, 'shift/discrete/images/val/front/images'), filter_weather=weather, filter_time=timeofday), metadata)


### 조건에 맞는 시퀀스 중 한 시퀀스만 랜덤하게 선택해서 return
# NO training set
def register_continuous_shift(image_root, dir_type, name, seq):
    metadata = {'image_root': image_root, 'stuff_classes': SHIFT_CATEGORIES, 'thing_classes': SHIFT_CATEGORIES, 'evaluator_type': "coco"}
    DatasetCatalog.register(name, lambda: load_shift_json(os.path.join(image_root, 'shift/{}/images/val/front/det_2d.json'.format(dir_type)),
                                                          os.path.join(image_root, 'shift/{}/images/val/front/images'.format(dir_type)),
                                                          single_folder=seq[0], filter_weather=None, filter_time=None))
    MetadataCatalog.get(name).set(**metadata)

 
