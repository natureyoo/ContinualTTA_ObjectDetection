# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import pprint
import sys
from collections.abc import Mapping

import wandb


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, Mapping) or not len(results), results
    logger = logging.getLogger(__name__)
    for task, res in results.items():
        if isinstance(res, Mapping):
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            logger.info("copypaste: Task: {}".format(task))
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
        else:
            logger.info(f"copypaste: {task}={res}")


def verify_results(cfg, results):
    """
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}

    Returns:
        bool: whether the verification succeeds or not
    """
    expected_results = cfg.TEST.EXPECTED_RESULTS
    if not len(expected_results):
        return True

    ok = True
    for task, metric, expected, tolerance in expected_results:
        actual = results[task].get(metric, None)
        if actual is None:
            ok = False
            continue
        if not np.isfinite(actual):
            ok = False
            continue
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False

    logger = logging.getLogger(__name__)
    if not ok:
        logger.error("Result verification failed!")
        logger.error("Expected Results: " + str(expected_results))
        logger.error("Actual Results: " + pprint.pformat(results))

        sys.exit(1)
    else:
        logger.info("Results verification passed.")
    return ok


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r


def plot_pr(pr_dicts, save_path, name="", dataset_size=10000):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 3, 1)
    precision = [pr_dicts["correct_class"][i] / pr_dicts["prediction"][i] if pr_dicts["prediction"][i] > 0 else 0 for i in range(100)]
    recall = [pr_dicts["correct_class"][i] / pr_dicts["gt"][i] if pr_dicts["gt"][i] > 0 else 0 for i in range(100)]
    plt.plot(np.arange(100) * 0.01, precision, c="blue", label="Precision")
    plt.plot(np.arange(100) * 0.01, recall, c="pink", label="Recall")
    plt.legend(loc='upper left')
    plt.xlabel('foreground score')
    plt.title("precision-recall of class level")

    plt.subplot(1, 3, 2)
    precision = [pr_dicts["correct_fg"][i] / pr_dicts["prediction"][i] if pr_dicts["prediction"][i] > 0 else 0 for i in range(100)]
    recall = [pr_dicts["correct_fg"][i] / pr_dicts["gt"][i] if pr_dicts["gt"][i] > 0 else 0 for i in range(100)]
    plt.plot(np.arange(100) * 0.01, precision, c="blue", label="Precision")
    plt.plot(np.arange(100) * 0.01, recall, c="pink", label="Recall")
    plt.legend(loc='upper left')
    plt.xlabel('foreground score')
    plt.title("precision-recall of fg-bg level")

    plt.subplot(1, 3, 3)
    pr_dicts["prediction"][0] = 0       # too big value, except for visualizing
    pred_cnts = [p / dataset_size for p in pr_dicts["prediction"]]
    plt.bar(np.arange(100) * 0.01, pred_cnts, 0.01)
    plt.xlabel('foreground score')
    plt.title("predicted foreground count per image")

    plt.suptitle(name)
    plt.savefig(save_path)


def plot_mh_dist(mh_distances, save_path, name=""):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8 * len(mh_distances), 8))
    for i, k in enumerate(mh_distances.keys()):
        plt.subplot(1, len(mh_distances), i + 1)
        mh = mh_distances[k].cpu().numpy()
        bin_size = (mh.max() - mh.min()) / 100
        x_ticks = [mh.min() + j * bin_size for j in range(100)]
        y_vals = [((mh >= x) & (mh < x + bin_size)).sum() for x in x_ticks]
        plt.plot(x_ticks, y_vals, c="blue")
        plt.xlabel('mahalanobis distance')
        plt.title("feature {}, {:.4f}".format(k, mh.mean()))
    plt.suptitle("mahalanobis distance distribution of {}".format(name))
    plt.savefig(save_path)

def plot_pr_by_mh(pr_dicts, mh_distances, save_path, name=""):
    import matplotlib.pyplot as plt

    # for pr_type in ["precision", "recall"]:
    #     plt.figure(figsize=(8 * len(mh_distances), 8 * len(pr_dicts)))
    #     for i, th in enumerate(pr_dicts.keys()):
    #         for j, k in enumerate(mh_distances.keys()):
    #             plt.subplot(len(pr_dicts), len(mh_distances), i * len(mh_distances) + j + 1)
    #             mh = mh_distances[k].cpu().numpy()
    #             plt.scatter(mh, pr_dicts[th][pr_type])
    #             plt.xlabel('mahalanobis distance')
    #             plt.ylabel(pr_type)
    #             plt.title("feature {}, {:.4f}, thr {}".format(k, mh.mean(), th))
    #     plt.suptitle("precision by mahalanobis dist of {}".format(name))
    #     plt.savefig("{}_{}.png".format(save_path, pr_type))

    plt.figure(figsize=(8 * len(mh_distances), 8 * 2))
    th = 0.5
    for i, pr_type in enumerate(["precision", "recall"]):
        for j, k in enumerate(mh_distances.keys()):
            ax1 = plt.subplot(2, len(mh_distances), i * len(mh_distances) + j + 1)
            mh = mh_distances[k].cpu().numpy()
            bin_size = (mh.max() - mh.min()) / 100
            x_ticks = [mh.min() + j * bin_size for j in range(100)]
            y_vals = [((mh >= x) & (mh < x + bin_size)).sum() for x in x_ticks]
            ax1.plot(x_ticks, y_vals, c="blue")
            ax1.set_ylabel("sample count")

            ax2 = ax1.twinx()
            y_vals2 = [np.array(pr_dicts[th][pr_type])[(mh >= x) & (mh < x + bin_size)].mean() for x in x_ticks]
            ax2.plot(x_ticks, y_vals2, c="deeppink", label=pr_type)
            ax2.set_ylabel(pr_type)

            plt.title("{}: feature {}, {:.4f}".format(pr_type, k, mh.mean()))
    plt.suptitle("precision and recall by mahalanobis dist of {}".format(name))
    plt.savefig("{}.png".format(save_path))

def plot_tsne(roi_heads, out_dir):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    import random

    features = roi_heads.selected_features.cpu()
    objectness = roi_heads.objectness.cpu()
    predictions = roi_heads.predictions.cpu()
    gt_labels = roi_heads.gts.cpu()

    colors = [(random.random(), random.random(), random.random()) for _ in range(roi_heads.num_classes+1)]
    fig = plt.figure(figsize=(60, 24))
    tsne_feat = TSNE(n_components=2).fit_transform(features)
    # plot per class of predictions
    ax = fig.add_subplot(121)
    for g in range(roi_heads.num_classes+1):
        ax.scatter(tsne_feat[predictions == g, 0], tsne_feat[predictions == g, 1], c=colors[g], label=g)
    ax.legend()
    ax.set_title("predictions")
    # plot per class of gt
    ax = fig.add_subplot(122)
    for g in range(roi_heads.num_classes+1):
        ax.scatter(tsne_feat[gt_labels == g, 0], tsne_feat[gt_labels == g, 1], c=colors[g], label=g)
    ax.legend()
    ax.set_title("gt labels")
    plt.savefig('{}/tsne_per_class.png'.format(out_dir))

    fig = plt.figure(figsize=(60, 24))
    # plot per obj of predictions
    ax = fig.add_subplot(121)
    idx = objectness >= 0.5
    ax.scatter(tsne_feat[idx, 0], tsne_feat[idx, 1], c="red", label="obj>0.5")
    idx = (objectness >= 0.3) & (objectness < 0.5)
    ax.scatter(tsne_feat[idx, 0], tsne_feat[idx, 1], c="blue", label="0.5>obj>0.3")
    idx = (objectness < 0.3)
    ax.scatter(tsne_feat[idx, 0], tsne_feat[idx, 1], c="green", label="0.3>obj>0.0")
    ax.legend()
    ax.set_title("predictions")

    # plot per obj of predictions
    ax = fig.add_subplot(122)
    idx = gt_labels != 80
    ax.scatter(tsne_feat[idx, 0], tsne_feat[idx, 1], c="red", label="object")
    idx = gt_labels == 80
    ax.scatter(tsne_feat[idx, 0], tsne_feat[idx, 1], c="green", label="background")
    ax.legend()
    ax.set_title("gt labels")
    plt.savefig('{}/tsne_per_obj.png'.format(out_dir))
