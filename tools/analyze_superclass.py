import torch
from torch.nn import functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 50})

coco_classes = [
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"dining table",
"toilet",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush",
"background"
]
coco_superclasses = [
    "person",   #1
    "vehicle","vehicle","vehicle","vehicle","vehicle","vehicle","vehicle","vehicle",    #8
    "outdoor","outdoor","outdoor","outdoor","outdoor",    # 5
    "animal","animal","animal","animal","animal","animal","animal","animal","animal","animal",   #10
    "accessory","accessory","accessory","accessory","accessory",    # 5
    "sports","sports","sports","sports","sports","sports","sports","sports","sports","sports",  # 10
    "kitchen","kitchen","kitchen","kitchen","kitchen","kitchen","kitchen",  # 7
    "food","food","food","food","food","food","food","food","food","food",  #10
    "furniture","furniture","furniture","furniture","furniture","furniture",    #6
    "electronic","electronic","electronic","electronic","electronic","electronic",  #6
    "appliance","appliance","appliance","appliance","appliance",    #5
    "indoor","indoor","indoor","indoor","indoor","indoor","indoor",  #7
    "background"
]
super_idx = [0,1,9,14,24,29,39,46,56,62,68,73,80]
super_nums = [1,8,5,10,5,10,7,10,6,6,5,7,1]
superclass_names = ["person","vehicle","outdoor","animal","accessory","sports","kitchen","food","furniture",
                    "electronic","appliance","indoor","background"]

num_classes = 81
s = torch.load("./models/coco_features_stat_0703.pth")
fg_feats = s['foreground']['source']['features']
fg_means = torch.stack([fg_feats[c].mean(dim=0) for c in range(num_classes)])
fg_cov = torch.stack([torch.cov(fg_feats[c].t()) for c in range(num_classes)])
cos_sim_mean = F.cosine_similarity(fg_means.unsqueeze(1), fg_means.unsqueeze(0), dim=2)

# plot cosine similarity
np_cos_sim_mean = np.asarray(cos_sim_mean.cpu())
fig = plt.figure(figsize=(100,100))
ax = fig.add_subplot(111)
cax = ax.matshow(np_cos_sim_mean)
fig.colorbar(cax)
# ax.set_xticks(np.arange(num_classes))
# ax.set_yticks(np.arange(num_classes))
# ax.set_xticklabels(['']+coco_superclasses)
# ax.set_yticklabels(['']+coco_superclasses)
ax.set_xticks(np.asarray(super_idx), superclass_names)
ax.set_yticks(np.asarray(super_idx), superclass_names)

plt.savefig("./vis/COCO/cosine_sim.png")

distributions = []
for c in range(num_classes):
    template_ext_cov = torch.eye(fg_means[c].shape[0]) * fg_cov[c].max().item() / 30
    cur_dist = torch.distributions.MultivariateNormal(fg_means[c], fg_cov[c] + template_ext_cov.to(fg_cov.device))
    distributions.append(cur_dist)

kl_div = []
for c_i in range(num_classes):
    cur_kl_div = []
    for c_j in range(num_classes):
        cur_kl_div.append(torch.distributions.kl.kl_divergence(distributions[c_i], distributions[c_j]))
    kl_div.append(torch.stack(cur_kl_div))

kl_div = np.asarray(torch.stack(kl_div).cpu())

# plot kl divergence
fig = plt.figure(figsize=(100,100))
ax = fig.add_subplot(111)
refined_kl_div = 1/(kl_div + kl_div[kl_div>0].min() * 5)
cax = ax.matshow(refined_kl_div)
fig.colorbar(cax)
# ax.set_xticks(np.arange(num_classes))
# ax.set_yticks(np.arange(num_classes))
# ax.set_xticklabels(['']+coco_superclasses)
# ax.set_yticklabels(['']+coco_superclasses)
ax.set_xticks(np.asarray(super_idx), superclass_names)
ax.set_yticks(np.asarray(super_idx), superclass_names)

plt.savefig("./vis/COCO/kl_div.png")
print('')
