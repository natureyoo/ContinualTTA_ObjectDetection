#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""

import logging
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import json

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format, lazy_inference_on_dataset_online_adaptation
from detectron2.utils import comm
from detectron2.layers import FrozenBatchNorm2d

from imagecorruptions import get_corruption_names


logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    results = {}
    if "evaluator" in cfg.dataloader:
        if not cfg.model.collect_features:
            for corrupt in get_corruption_names()[:]:
                # cfg.dataloader.test.corrupt = corrupt
                cfg.dataloader.test.dataset.names = "coco_2017_val-{}".format(corrupt)
                ret, _ = inference_on_dataset(
                    model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                )
                print_csv_format(ret)
                results[corrupt] = ret
        cfg.dataloader.test.dataset.names = "coco_2017_val"
        ret, _ = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        results['original'] = ret
        json.dump(results, os.path.join(cfg.train.output_dir, 'results.json', 'w'))
        return results


def do_adapt(cfg, model, where='backbone'):
    # configure learnable part of model and set optimizer
    params = []
    bn_params = []
    model.eval()
    model.requires_grad_(False)
    if where == 'backbone':
        for m_name, m in model.backbone.bottom_up.named_modules():
            #if cfg.TEST.ADAPTATION.NORM == 'adapt':
            if True:
                if isinstance(m, FrozenBatchNorm2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    m.adapt_training = False
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
                    if m.weight.requires_grad:
                        #bn_params += [m.weight, m.bias]
                        params += [m.weight, m.bias]
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                # except patch embedding
                if "patch_embed" in m_name and "attn" in m_name:
                    continue
                m.weight.requires_grad = True
                params += [m.weight]
                if m.bias is not None:
                    m.bias.requires_grad = True
                    params += [m.bias]
    elif where == 'adapter':
        for stage in model.backbone.bottom_up.stages:
            for block in stage:
                block.adapter.requires_grad_(True)
                params += list(block.adapter.parameters())
    sgd_args = [{"params": params, 'lr': 0.001}]
    if len(bn_params) > 0:
        sgd_args.append({"params": bn_params, "lr": cfg.SOLVER.BASE_LR_BN})
    optimizer = torch.optim.SGD(sgd_args)
    # optimizer = maybe_add_gradient_clipping(cfg, optimizer)

    results = {}
    loss_ema99 = 0.0
    if "evaluator" in cfg.dataloader:
        for corrupt in get_corruption_names()[:]:
            # cfg.dataloader.test.corrupt = corrupt
            cfg.dataloader.test.dataset.names = "coco_2017_val-{}".format(corrupt)
            ret, loss_ema99 = lazy_inference_on_dataset_online_adaptation(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator), optimizer, loss_ema99=loss_ema99
            )
            print_csv_format(ret)
            results[corrupt] = ret
        cfg.dataloader.test.dataset.names = "coco_2017_val"
        ret, _ = lazy_inference_on_dataset_online_adaptation(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator), optimizer, loss_ema99=loss_ema99
        )
        print_csv_format(ret)
        results['original'] = ret
        json.dump(results, open(os.path.join(cfg.train.output_dir, 'results.json'), 'w'))
        return results


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        if cfg.model.online_adapt:
            print(do_adapt(cfg, model, where='adapter'))
        else:
            print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
