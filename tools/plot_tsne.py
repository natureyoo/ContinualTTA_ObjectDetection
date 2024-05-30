import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random, os
import torch
from torch.nn import functional as F
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import CosineSimilarity
# plt.ion()

dir_path = {"train": "./outputs/COCO/Source_Train_Features_0903/",
         "source": "./outputs/COCO/r50_fpn_backbone_features_0828/",
         "gaussian_noise": "./outputs/COCO/gaussian_noise_global_fg_adapted_model",
         "shot_noise": "./outputs/COCO/gaussian_noise_global_fg_adapted_model",
         }

# feats_path = "./outputs/COCO/r50_fpn_backbone_features_0828/features_stat.pth"
features = {}

# domain_labels = list(features.keys())
domain_labels = ["train", "source", "gaussian_noise", "shot_noise"]
# domain_labels = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "source"]

for domain in domain_labels:
    features[domain] = torch.load(os.path.join(dir_path[domain], "features_stat_{}.pth".format(domain)))["backbone"]

level_keys = list(features["source"].keys())
per_domain_num = features["source"][level_keys[0]][2].shape[0]
cluster_models = {}
### analysis
for l_idx, level in enumerate(level_keys):
    feats0 = features["train"][level][2]
    # feats1 = features["gaussian_noise"][level][2]
    # u0, s0, v0 = torch.svd(feats0)
    # u1, s1, v1 = torch.svd(feats1)
    # dist0 = F.cosine_similarity(feats0.unsqueeze(1), v0.unsqueeze(0), dim=2)
    # dist1 = F.cosine_similarity(feats1.unsqueeze(1), v0.unsqueeze(0), dim=2)
    cluster_model = KMeans(n_clusters=10, distance=CosineSimilarity, normalize='unit')
    cluster_model = cluster_model.fit(feats0.unsqueeze(0))
    cluster_models[level] = cluster_model
    # labels0 = cluster_model.predict(feats0.unsqueeze(0))
    # labels1 = cluster_model.predict(feats1.unsqueeze(0))
###

colors = [(random.random(), random.random(), random.random()) for _ in domain_labels]
fig = plt.figure(figsize=(60, 24))

for l_idx, level in enumerate(level_keys):
    print("drawing {}-th level features...".format(level))
    cur_feats = torch.cat([features[k][level][2] for k in features])
    tsne_feat = TSNE(n_components=2).fit_transform(cur_feats.cpu())
    # plot per class of predictions
    ax = fig.add_subplot(231+l_idx)
    for d_idx, domain in enumerate(domain_labels):
        ax.scatter(tsne_feat[d_idx * per_domain_num: (d_idx + 1) * per_domain_num, 0], tsne_feat[d_idx * per_domain_num: (d_idx + 1) * per_domain_num, 1], c=colors[d_idx], label=domain)
    ax.legend()
    ax.set_title("{}-th level backbone features".format(level))

plt.savefig(os.path.join(dir_path["gaussian_noise"], 'TSNE_backbone_feats_{}.png'.format('-'.join(domain_labels))))
