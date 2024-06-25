from PIL import Image
import os
import xml.etree.ElementTree as ET

root_dir = "/home/jayeon/Data/Cross-Domain-Detection/clipart"
img_dir = "JPEGImages"
ann_dir = "Annotations"

img_list = os.listdir(os.path.join(root_dir, img_dir))
for img_file in img_list:
    im = Image.open(os.path.join(root_dir, img_dir, img_file))
    ann = ET.parse(open(os.path.join(root_dir, ann_dir, img_file.replace('.jpg', '.xml'))))
    root = ann.getroot()
    print('')