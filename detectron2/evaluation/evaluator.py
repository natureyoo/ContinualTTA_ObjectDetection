# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
import numpy as np
import os
from detectron2.solver import adjust_learning_rate
from detectron2.structures import Instances
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.events import EventStorage

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.visualizer import Visualizer


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def extract_object_patch(cfg, data_loader, domain_name):
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    obj_queue = {k: [] for k in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}
    patch_num = 20
    for idx, inputs in enumerate(data_loader):
        for inp in inputs:
            cur_classes = [0 for _ in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)]
            objects = inp['instances']
            for i in range(len(objects)):
                obj_region = objects[i].gt_boxes.tensor.reshape(-1)
                cropped_region = inp['image'][:, obj_region[1].int():obj_region[3].int()+1,
                             obj_region[0].int():obj_region[2].int()+1]
                if cropped_region.shape[1] * cropped_region.shape[2] > 128 and cur_classes[objects[i].gt_classes.item()] == 0 and len(obj_queue[objects[i].gt_classes.item()]) < patch_num:
                    cur_classes[objects[i].gt_classes.item()] += 1
                    obj_queue[objects[i].gt_classes.item()].append(cropped_region)
        print("current {} classes, minimum {} samples".format(sum([len(obj_queue[k]) >= patch_num for k in obj_queue]), min([len(obj_queue[k]) for k in obj_queue])))
        if sum([len(obj_queue[k]) >= patch_num  for k in obj_queue]) == cfg.MODEL.ROI_HEADS.NUM_CLASSES:
            break


        # obj_region = inst_pred_boxes.tensor[i]
        # if (obj_region[2] - obj_region[0]) * (obj_region[3] - obj_region[1]) > 256:
        #     pseudo_gt_boxes.append(obj_region)
        #     pseudo_gt_classes.append(inst.pred_classes[i])
        #
        #     cropped_region = input_['image'][:, obj_region[1].int():obj_region[3].int(),
        #                      obj_region[0].int():obj_region[2].int()]
        #     if cropped_region.shape[1] > 0 and cropped_region.shape[2] > 0:
        #         obj_queue[inst.pred_classes[i].item()]["obj"].append(cropped_region)
        #         if len(obj_queue[inst.pred_classes[i].item()]["obj"]) > 5:
        #             random.shuffle(obj_queue[inst.pred_classes[i].item()]["obj"])
        #             obj_queue[inst.pred_classes[i].item()]["obj"] = obj_queue[inst.pred_classes[i].item()]["obj"][:5]
    torch.save(obj_queue, os.path.join(cfg.OUTPUT_DIR, "obj_patches_{}.pt".format(domain_name)))

def plot_pr_stat(cnt, values, file_name):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(36, 24))
    ## precision
    ax = fig.add_subplot(231)
    ### with background
    ax.plot(np.arange(0, 1, 0.01), cnt['correct_with_bg'] / cnt['total'], c='b', label='Precision')
    ### without background
    ax.plot(np.arange(0, 1, 0.01), cnt['correct_with_bg'] / cnt['gt'], c='r', label='Recall')
    ax.legend()
    ax.set_title("With Background")

    ## recall
    ax = fig.add_subplot(232)
    ### with background
    ax.plot(np.arange(0, 1, 0.01), cnt['correct_without_bg'] / cnt['total'], c='b', label='Precision')
    ### without background
    ax.plot(np.arange(0, 1, 0.01), cnt['correct_without_bg'] / cnt['gt'], c='r', label='Recall')
    ax.legend()
    ax.set_title("Without Background")

    ## recall
    # ax = fig.add_subplot(233)
    # ### with background
    # ax.plot(np.arange(0, 1, 0.01), cnt['correct_intra'] / cnt['total'], c='b', label='Precision')
    # ### without background
    # ax.plot(np.arange(0, 1, 0.01), cnt['correct_intra'] / cnt['gt'], c='r', label='Recall')
    # ax.legend()
    # ax.set_title("Target Prototype")

    ax = fig.add_subplot(234)
    # sampling
    bg_score = torch.Tensor([])
    max_score_with_bg = torch.Tensor([])
    max_score_without_bg = torch.Tensor([])
    entropy_without_bg = torch.Tensor([])
    for c in np.arange(80):
        target_idx = (values['gt_label']==c) & (values['preds'] == c)
        idx = torch.where(target_idx)[0][torch.randperm(target_idx.sum())[:1000]]
        bg_score = torch.cat([bg_score, values['bg_score'][idx]])
        max_score_with_bg = torch.cat([max_score_with_bg, values['max_score_with_bg'][idx]])
        max_score_without_bg = torch.cat([max_score_without_bg, values['max_score_without_bg'][idx]])
        entropy_without_bg = torch.cat([entropy_without_bg, values['entropy_without_bg'][idx]])
    ax.scatter(bg_score,
               entropy_without_bg, c='g', label='foreground-correct')
    bg_score2 = torch.Tensor([])
    max_score_with_bg2 = torch.Tensor([])
    max_score_without_bg2 = torch.Tensor([])
    entropy_without_bg2 = torch.Tensor([])
    for c in np.arange(80):
        target_idx = (values['gt_label']==c) & (values['preds'] != c)
        idx = torch.where(target_idx)[0][torch.randperm(target_idx.sum())[:1000]]
        bg_score2 = torch.cat([bg_score2, values['bg_score'][idx]])
        max_score_with_bg2 = torch.cat([max_score_with_bg2, values['max_score_with_bg'][idx]])
        max_score_without_bg2 = torch.cat([max_score_without_bg2, values['max_score_without_bg'][idx]])
        entropy_without_bg2 = torch.cat([entropy_without_bg2, values['entropy_without_bg'][idx]])
    ax.scatter(bg_score2,
               entropy_without_bg2, c='b', label='foreground-wrong')
    c=80
    idx = torch.where(values['gt_label'] == c)[0][torch.randperm((values['gt_label'] == c).sum())[:50000]]
    bg_score3 = values['bg_score'][idx]
    max_score_with_bg3 = values['max_score_with_bg'][idx]
    max_score_without_bg3 = values['max_score_without_bg'][idx]
    entropy_without_bg3 = values['entropy_without_bg'][idx]
    ax.scatter(bg_score3, entropy_without_bg3, c='r', label='background')
    ax.legend()
    ax.set_title("Entropy Without BG vs. BG score")

    ax = fig.add_subplot(235)
    ax.scatter(bg_score,
               max_score_without_bg, c='g', label='foreground-correct')
    ax.scatter(bg_score2,
               max_score_without_bg2, c='b', label='foreground-wrong')
    ax.scatter(bg_score3, max_score_without_bg3, c='r', label='background')
    ax.legend()
    ax.set_title("Max Score Without BG vs. BG score")
    plt.savefig(file_name)


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], collect_features=False, domain_name=None, visualize_dir=None
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    num_samples = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0
            if collect_features and num_samples > 5000:
                break
            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            # visualize
            if visualize_dir is not None:
                os.makedirs(visualize_dir, exist_ok=True)
                for inp, oup in zip(inputs, outputs):
                    img = cv2.imread(inp["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
                    basename = os.path.basename(inp["file_name"])
                #
                #     # create instances
                    ret = Instances(img.shape[:2])
                #
                    score = np.asarray(oup['instances'].scores.cpu())
                    chosen = (score > 0.5).nonzero()[0]
                    score = score[chosen]
                    bbox = np.asarray(oup['instances'].pred_boxes.tensor.cpu()[chosen]).reshape(-1, 4)
                    bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYXY_ABS)
                #
                    labels = np.asarray(oup['instances'].pred_classes.cpu()[chosen])
                #
                    ret.scores = score
                    ret.pred_boxes = Boxes(bbox)
                    ret.pred_classes = labels
                #
                    metadata = MetadataCatalog.get('kitti_val')
                    vis = Visualizer(img, metadata)
                    vis_pred = vis.draw_instance_predictions(ret).get_image()
                #
                    vis = Visualizer(img, metadata)
                    # vis_gt = vis.draw_dataset_dict(inp).get_image()
                    # vis_gt = vis.draw_dataset_input(inp, vis_pred.shape[:2], img.shape[:2]).get_image()    # , output_size, image_size
                    vis_gt = vis.draw_dataset_input(inp, vis_pred.shape[:2], (inp['height'], inp['width'])).get_image()    # , output_size, image_size
                #
                    concat = np.concatenate((vis_pred, vis_gt), axis=1)
                    cv2.imwrite(os.path.join(visualize_dir, basename), concat[:, :, ::-1])

            num_samples += len(inputs)
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate(domain_name=domain_name)
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results, total_compute_time


def set_pseudo_labels(inputs, outputs, conf_th=0.5):
    new_inputs = []
    for inp, oup in zip(inputs, outputs):
        inst = oup['instances'][oup['instances'].scores > conf_th]
        new_inp = {k: inp[k] for k in inp if k not in ['instances', 'image', 'strong_aug_image']}
        new_inp['image'] = inp['strong_aug_image']
        new_img_size, ori_img_size = inp['instances'].image_size, inst.image_size
        new_inst = Instances(new_img_size)
        new_inst.gt_classes = inst.pred_classes
        new_inst.gt_boxes = inst.pred_boxes
        new_inst.gt_boxes.scale(new_img_size[1] / ori_img_size[1], new_img_size[0] / ori_img_size[0])
        new_inp['instances'] = new_inst
        new_inputs.append(new_inp)
    return new_inputs

def inference_on_dataset_online_adaptation(cfg, model, data_loader, optimizer, evaluator, d_idx, wandb, teacher_model=None, val_data_loader=None, val_evaluator=None, loss_ema99=0, loss_ema95=0, loss_ema90=0, is_used=0, domain_name=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    # logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()
    if val_evaluator is not None:
        val_evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    batch_size = cfg.SOLVER.IMS_PER_BATCH_TEST
    cur_used = True
    prev_used = is_used
    f_sim = {}
    div_thr = 2* sum(model.s_div.values()) * cfg.TEST.ADAPTATION.SKIP_TAU if cfg.TEST.ADAPTATION.SKIP_REDUNDANT is not None else 2* sum(model.s_div.values())
    # for weight regularization
    init_weights = []
    for p_idx, _p in enumerate(optimizer.param_groups):
        p = _p['params'][0]
        init_weights.append(p.clone().detach())

    with EventStorage() as storage:
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            if idx == total:
                break

            start_compute_time = time.perf_counter()

            cur_step = (d_idx * len(data_loader) + idx) * batch_size
            #if idx / len(data_loader) < cfg.TEST.ADAPTATION.STOP:\
            if cur_used or (not cur_used and pause_iter % cfg.TEST.ADAPTATION.SKIP_PERIOD == 0) or ('period' in cfg.TEST.ADAPTATION.SKIP_REDUNDANT and idx % cfg.TEST.ADAPTATION.SKIP_PERIOD == 0):
                if cfg.TEST.ADAPTATION.TYPE == "mean-teacher":
                    outputs = teacher_model(inputs)
                    new_inputs = set_pseudo_labels(inputs, outputs, conf_th=cfg.TEST.ADAPTATION.TH_BG)
                    losses = model(new_inputs)
                elif cfg.TEST.ADAPTATION.TYPE == "mean-teacher-align":
                    outputs = teacher_model(inputs)
                    new_inputs = set_pseudo_labels(inputs, outputs, conf_th=cfg.TEST.ADAPTATION.TH_BG)
                    model.online_adapt = False
                    model.proposal_generator.training = True
                    model.roi_heads.training = True
                    detector_losses = model(new_inputs)
                    model.online_adapt = True
                    outputs, losses = model(inputs)
                    losses.update(detector_losses)
                else:
                    outputs, losses, feature_sim = model(inputs)

                # weight regularization
                if cfg.TEST.ADAPTATION.WEIGHT_REG > 0.0:
                    stick_loss = 0
                    for p_idx, (_p, s) in enumerate(zip(optimizer.param_groups, init_weights)):
                        p = _p['params'][0]
                        stick_loss += torch.mean((p - s) ** 2)
                    losses["stick"] = cfg.TEST.ADAPTATION.WEIGHT_REG * stick_loss
                total_loss = sum([losses[k] for k in losses])
                #not_redundant = min([feature_sim[k] for k in feature_sim if 'gl' in k]) < cfg.TEST.ADAPTATION.SKIP_THRESHOLD if cfg.TEST.ADAPTATION.SKIP_REDUNDANT else True
                #cur_used = losses["global_align"] > div_thr or not_redundant
                #cur_used = losses["global_align"] > div_thr or idx % cfg.TEST.ADAPTATION.SKIP_PERIOD == 0
                cur_used = False
                if cfg.TEST.ADAPTATION.SKIP_REDUNDANT is None:
                    cur_used = True
                elif 'stat' in cfg.TEST.ADAPTATION.SKIP_REDUNDANT and losses["global_align"] > div_thr:
                    cur_used = True
                elif 'period' in cfg.TEST.ADAPTATION.SKIP_REDUNDANT and idx % cfg.TEST.ADAPTATION.SKIP_PERIOD == 0:
                    cur_used = True
                elif 'ema' in cfg.TEST.ADAPTATION.SKIP_REDUNDANT and losses["global_align"] / (loss_ema99 + 1e-7) > cfg.TEST.ADAPTATION.SKIP_BETA:
                    cur_used = True
                # cur_used = losses["global_align"] / (loss_ema99 + 1e-7) > cfg.TEST.ADAPTATION.SKIP_BETA if cfg.TEST.ADAPTATION.SKIP_REDUNDANT else True
                is_used += int(cur_used)
                if total_loss > 0 and cur_used:
                    total_loss.backward()
                    if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                        torch.nn.utils.clip_grad_norm_(model.backbone.parameters(),
                                                       cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE)
                    optimizer.step()
                else:
                    pause_iter = 1
                optimizer.zero_grad()
                # EMA update
                if cfg.TEST.ADAPTATION.TYPE is not None and "mean-teacher" in cfg.TEST.ADAPTATION.TYPE:
                    beta = cfg.TEST.ADAPTATION.EMA_BETA
                    for t_p, s_p in zip(teacher_model.parameters(), model.parameters()):
                        if s_p.requires_grad:
                            t_p.data = beta * t_p.data + (1 - beta) * s_p.data
                loss_str = " ".join(["{}: {:.6f}, ".format(k, losses[k].item()) for k in losses])
                if wandb is not None:
                    wandb.log(losses, step=cur_step)
                    wandb.log({"total_loss": total_loss}, step=cur_step)
                    #wandb.log(feature_sim, step=cur_step)
                    wandb.log({"accumulated_used": is_used}, step=cur_step)
                    wandb.log({"cur_used": is_used - prev_used}, step=cur_step)
                    if "global_align" in losses:
                        wandb.log({"ema99_ratio": losses["global_align"].item() / (loss_ema99 + 1e-7)}, step=cur_step)
                        wandb.log({"ema95_ratio": losses["global_align"].item() / (loss_ema95 + 1e-7)}, step=cur_step)
                        wandb.log({"ema90_ratio": losses["global_align"].item() / (loss_ema90 + 1e-7)}, step=cur_step)
                        wandb.log({"kl_div_ratio": losses["global_align"].item() / div_thr}, step=cur_step)
                if "global_align" in losses:
                    loss_ema99 = 0.99 * loss_ema99 + 0.01 * losses["global_align"].item()
                    loss_ema95 = 0.95 * loss_ema95 + 0.05 * losses["global_align"].item()
                    loss_ema90 = 0.9 * loss_ema90 + 0.1 * losses["global_align"].item()
                del losses, total_loss
            else:
                with torch.no_grad():
                    outputs = model.inference(inputs)
                loss_str = ""
                pause_iter += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_compute_time += time.perf_counter() - start_compute_time

            with torch.no_grad():
                evaluator.process(inputs, outputs)

            if val_data_loader is not None and idx % 50 == 0:
                model.online_adapt = False
                val_results, _ = inference_on_dataset(model, val_data_loader, val_evaluator)
                if wandb is not None:
                    wandb.log({'val-mAP': val_results['bbox']['AP'], 'val-mAP50': val_results['bbox']['AP50']}, step=cur_step)
                model.online_adapt = True

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                print_str = "Inference done {}/{}. {:.4f} s / img. ETA={} ".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                print_str += loss_str
                print_str += "lr: {}".format(optimizer.param_groups[0]['lr'])
                log_every_n_seconds(
                    logging.INFO,
                    print_str,
                    n=5,
                )
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate(domain_name=domain_name)
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle

    if results is None:
        results = {}
    return results, loss_ema99, loss_ema95, loss_ema90, is_used, total_compute_time


# model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], collect_features=False, domain_name=None, visualize_dir=None
def lazy_inference_on_dataset_online_adaptation(model, data_loader, evaluator, optimizer, domain_name=None, loss_ema99=0.0):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    # logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    # batch_size = cfg.SOLVER.IMS_PER_BATCH_TEST
    cur_used = True
    skip_period = 10
    skip_type = None #'stat-ema'
    adaptation_type = 'ours'
    is_used = 0
    # prev_used = is_used
    f_sim = {}
    # div_thr = 2* sum(model.s_div.values()) * cfg.TEST.ADAPTATION.SKIP_TAU if cfg.TEST.ADAPTATION.SKIP_REDUNDANT is not None else 2* sum(model.s_div.values())
    # for weight regularization
    init_weights = []
    # for p_idx, _p in enumerate(optimizer.param_groups):
    #     p = _p['params'][0]
    #     init_weights.append(p.clone().detach())

    with EventStorage() as storage:
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            if idx == total:
                break

            start_compute_time = time.perf_counter()

            # cur_step = (d_idx * len(data_loader) + idx) * batch_size
            #if idx / len(data_loader) < cfg.TEST.ADAPTATION.STOP:\
            if cur_used or (not cur_used and pause_iter % skip_period == 0) or ('period' in skip_type and skip_period == 0):
                outputs, losses = model(inputs)
                # if adaptation_type == "mean-teacher":
                #     outputs = teacher_model(inputs)
                #     new_inputs = set_pseudo_labels(inputs, outputs, conf_th=cfg.TEST.ADAPTATION.TH_BG)
                #     losses = model(new_inputs)
                # elif adaptation_type == "mean-teacher-align":
                #     outputs = teacher_model(inputs)
                #     new_inputs = set_pseudo_labels(inputs, outputs, conf_th=cfg.TEST.ADAPTATION.TH_BG)
                #     model.online_adapt = False
                #     model.proposal_generator.training = True
                #     model.roi_heads.training = True
                #     detector_losses = model(new_inputs)
                #     model.online_adapt = True
                #     outputs, losses = model(inputs)
                #     losses.update(detector_losses)
                # else:
                #     outputs, losses = model(inputs)

                total_loss = sum([losses[k] for k in losses])
                #not_redundant = min([feature_sim[k] for k in feature_sim if 'gl' in k]) < cfg.TEST.ADAPTATION.SKIP_THRESHOLD if cfg.TEST.ADAPTATION.SKIP_REDUNDANT else True
                #cur_used = losses["global_align"] > div_thr or not_redundant
                #cur_used = losses["global_align"] > div_thr or idx % cfg.TEST.ADAPTATION.SKIP_PERIOD == 0
                cur_used = False
                if skip_type is None:
                    cur_used = True
                # elif 'stat' in skip_type and losses["global_align"] > div_thr:
                #     cur_used = True
                # elif 'period' in skip_type and idx % cfg.TEST.ADAPTATION.SKIP_PERIOD == 0:
                #     cur_used = True
                # elif 'ema' in skip_type and losses["global_align"] / (loss_ema99 + 1e-7) > cfg.TEST.ADAPTATION.SKIP_BETA:
                #     cur_used = True
                # cur_used = losses["global_align"] / (loss_ema99 + 1e-7) > cfg.TEST.ADAPTATION.SKIP_BETA if cfg.TEST.ADAPTATION.SKIP_REDUNDANT else True
                is_used += int(cur_used)
                if total_loss > 0 and cur_used:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
                    optimizer.step()
                else:
                    pause_iter = 1
                optimizer.zero_grad()
                # EMA update
                # if cfg.TEST.ADAPTATION.TYPE is not None and "mean-teacher" in cfg.TEST.ADAPTATION.TYPE:
                #     beta = cfg.TEST.ADAPTATION.EMA_BETA
                #     for t_p, s_p in zip(teacher_model.parameters(), model.parameters()):
                #         if s_p.requires_grad:
                #             t_p.data = beta * t_p.data + (1 - beta) * s_p.data
                loss_str = " ".join(["{}: {:.6f}, ".format(k, losses[k].item()) for k in losses])
                # if wandb is not None:
                #     wandb.log(losses, step=cur_step)
                #     wandb.log({"total_loss": total_loss}, step=cur_step)
                #     #wandb.log(feature_sim, step=cur_step)
                #     wandb.log({"accumulated_used": is_used}, step=cur_step)
                #     wandb.log({"cur_used": is_used - prev_used}, step=cur_step)
                #     if "global_align" in losses:
                #         wandb.log({"ema99_ratio": losses["global_align"].item() / (loss_ema99 + 1e-7)}, step=cur_step)
                #         wandb.log({"ema95_ratio": losses["global_align"].item() / (loss_ema95 + 1e-7)}, step=cur_step)
                #         wandb.log({"ema90_ratio": losses["global_align"].item() / (loss_ema90 + 1e-7)}, step=cur_step)
                #         wandb.log({"kl_div_ratio": losses["global_align"].item() / div_thr}, step=cur_step)
                if "global_align" in losses:
                    loss_ema99 = 0.99 * loss_ema99 + 0.01 * losses["global_align"].item()
                del losses, total_loss
            else:
                with torch.no_grad():
                    outputs = model.inference(inputs)
                loss_str = ""
                pause_iter += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_compute_time += time.perf_counter() - start_compute_time

            with torch.no_grad():
                evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                print_str = "Inference done {}/{}. {:.4f} s / img. ETA={} ".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                print_str += loss_str
                print_str += "lr: {}".format(optimizer.param_groups[0]['lr'])
                log_every_n_seconds(
                    logging.INFO,
                    print_str,
                    n=5,
                )
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate(domain_name=domain_name)
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle

    if results is None:
        results = {}
    # return results, loss_ema99, loss_ema95, loss_ema90, is_used, total_compute_time
    return results, loss_ema99


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
