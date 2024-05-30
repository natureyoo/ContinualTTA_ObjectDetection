from imagecorruptions import corrupt
from PIL import Image
import numpy as np
import os

base_dir = "/home/jayeon/Data/KITTI/data_object_label_2/training/image_2"
corruption_name = 'snow'
corruption_dir = "{}_{}".format(base_dir, corruption_name)
os.makedirs(corruption_dir, exist_ok=True)

for idx, img in enumerate(os.listdir(base_dir)):
    image = np.asarray(Image.open(os.path.join(base_dir, img)))
    corrupted_image = corrupt(image, corruption_name=corruption_name, severity=5)
    corrupted_image = Image.fromarray(corrupted_image)
    corrupted_image.save(os.path.join(corruption_dir, img))
    if idx % 1000 == 0:
        print("{}-th image processing...".format(idx))
