import cv2
import torch


@classmethod
def test_online_adaptation(cls, cfg, wandb=None, wandb_step_size=0, evaluators=None):
    """
    Evaluate the given model. The given model is expected to already contain
    weights to evaluate.

    Args:
        cfg (CfgNode):
        evaluators (list[DatasetEvaluator] or None): if None, will call
            :meth:`build_evaluator`. Otherwise, must have the same length as
            ``cfg.DATASETS.TEST``.

    Returns:
        dict: a dict of result metrics
    """
    logger = logging.getLogger(__name__)
    if isinstance(evaluators, DatasetEvaluator):
        evaluators = [evaluators]
    if evaluators is not None:
        assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
            len(cfg.DATASETS.TEST), len(evaluators)
        )

    step_size = wandb_step_size
    results = OrderedDict()
    # backbone_feats = {}
    # foreground_feats = {}
    if cfg.TEST.CORRUPT:
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            for c_idx, corrupt in enumerate(get_corruption_names()[:]):
                if cfg.TEST.ADAPTED_MODEL is not None:
                    weight_path = os.path.join(cfg.TEST.ADAPTED_MODEL, "{}_adapted_model.pth".format(corrupt))
                else:
                    weight_path = None
                model, _ = configure_model(cfg, DefaultTrainer, revert=True, weight_path=weight_path)

                data_loader = cls.build_test_loader(cfg, "{}-{}".format(dataset_name, corrupt))
                # When evaluators are passed in as arguments,
                # implicitly assume that evaluators can be created before data_loader.
                if evaluators is not None:
                    evaluator = evaluators[idx]
                else:
                    try:
                        evaluator = cls.build_evaluator(cfg, dataset_name)
                        evaluators_iter = cls.build_evaluator(cfg, dataset_name)
                    except NotImplementedError:
                        logger.warn(
                            "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                            "or implement its `build_evaluator` method."
                        )
                        results[dataset_name] = {}
                        continue
                results_i = inference_on_dataset(cfg, model, data_loader, evaluator, c_idx, wandb, evaluator_iter=None)
                results["{}-{}".format(dataset_name, corrupt)] = results_i
                if comm.is_main_process():
                    assert isinstance(
                        results_i, dict
                    ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                    logger.info(
                        "Evaluation results for {} in csv format:".format("{}-{}".format(dataset_name, corrupt)))
                    print_csv_format(results_i)
                if wandb is not None:
                    wandb.log({"mAP": results_i['bbox']['AP'], "mAP50": results_i['bbox']['AP50']},
                              step=int((c_idx + 1) * len(data_loader) * cfg.SOLVER.IMS_PER_BATCH_TEST))

                if cfg.TEST.OUT_FEATURES:
                    # backbone_feats[corrupt] = {}
                    # # for k in model.backbone_features.keys():
                    # for k in cfg.MODEL.RPN.IN_FEATURES:
                    #     feats = model.backbone_features[k]
                    #     mean = feats.mean(dim=0)
                    #     cov = (feats - mean).t() @ (feats - mean)
                    #     backbone_feats[corrupt][k] = (mean, cov, feats)
                    # model.backbone_features = {}
                    backbone_feats = {}
                    for k in model.backbone_features:
                        feats = model.backbone_features[k]
                        mean = feats.mean(dim=0)
                        # cov = (feats - mean).t() @ (feats - mean)
                        cov = torch.cov(feats.t())
                        # var = torch.var(feats, dim=0, keepdim=True)
                        backbone_feats[k] = (mean, cov, feats)
                        # backbone_feats[k] = (mean, var)

                    out_feats = {"backbone": backbone_feats}
                    torch.save(out_feats, os.path.join(cfg.OUTPUT_DIR, "features_stat_{}.pth".format(corrupt)))
                    # foreground features
                    # foreground_feats[corrupt] = {"features": model.roi_heads.fg_features, "gt": model.roi_heads.iou_with_gt}

                # if cfg.TEST.ANALYSIS:
                #     dataset_size = len(data_loader) * cfg.SOLVER.IMS_PER_BATCH_TEST
                #     plot_pr(model.roi_heads.pr, os.path.join(cfg.OUTPUT_DIR, "PR_{}.png".format(corrupt)), corrupt, dataset_size)
                #     plot_mh_dist(model.mh_distance, os.path.join(cfg.OUTPUT_DIR, "MH_distance_{}.png".format(corrupt)), corrupt)
                #     plot_pr_by_mh(model.roi_heads.pr_per_img, model.mh_distance,
                #                  os.path.join(cfg.OUTPUT_DIR, "PR_by_MH_{}".format(corrupt)), corrupt)

    step = c_idx + 2 if cfg.TEST.CORRUPT else 1
    for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        # model, _ = configure_model(cfg, DefaultTrainer, revert=True)
        data_loader = cls.build_test_loader(cfg, dataset_name)
        # When evaluators are passed in as arguments,
        # implicitly assume that evaluators can be created before data_loader.
        if evaluators is not None:
            evaluator = evaluators[idx]
        else:
            try:
                evaluator = cls.build_evaluator(cfg, dataset_name)
                evaluators_iter = cls.build_evaluator(cfg, dataset_name)
            except NotImplementedError:
                logger.warn(
                    "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                    "or implement its `build_evaluator` method."
                )
                results[dataset_name] = {}
                continue
        results_i = inference_on_dataset(cfg, model, data_loader, evaluator, step - 1, wandb, evaluator_iter=None)
        results[dataset_name] = results_i
        if comm.is_main_process():
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        backbone_feats = {}
        for k in model.backbone_features:
            feats = model.backbone_features[k]
            mean = feats.mean(dim=0)
            # cov = (feats - mean).t() @ (feats - mean)
            cov = torch.cov(feats.t())
            backbone_feats[k] = (mean, cov, feats)
            # var = torch.var(feats, dim=0, keepdim=True)
            # backbone_feats[k] = (mean, var)
        backbone_feats = {"source": {5000: backbone_feats}}
        out_feats = {"backbone": backbone_feats}
        foreground_feats = {"features": model.roi_heads.fg_features, "gt": model.roi_heads.iou_with_gt}
        torch.save(out_feats, os.path.join(cfg.OUTPUT_DIR, "features_stat_swinT_0929.pth"))
    if wandb is not None:
        wandb.log({"mAP": results_i['bbox']['AP'], "mAP50": results_i['bbox']['AP50']},
                  step=int(step * len(data_loader) * cfg.SOLVER.IMS_PER_BATCH_TEST))

    # if cfg.TEST.OUT_FEATURES:
    #     out_feats = {}
    #     # backbone features
    #     backbone_feats["source"] = {}
    #     sample_nums = [500, 5000]
    #     for n in sample_nums:
    #         backbone_feats["source"][n] = {}
    #     # for k in model.backbone_features.keys():
    #     for k in ["p2", "p3", "p4", "p5", "p6"]:
    #         feats = model.backbone_features[k]
    #         for n in sample_nums:
    #             mean = feats[:n].mean(dim=0)
    #             # cov = (feats - mean).t() @ (feats - mean)
    #             cov = torch.cov(feats[:n].t())
    #             backbone_feats["source"][n][k] = (mean, cov, feats)
    #     # save style statistics of 500 source domain samples
    #     # backbone_feats["source-style"] = {}
    #     # for k in ["stem", "res2", "res3", "res4", "res5"]:
    #     #     backbone_feats["source-style"][k] = model.backbone_features[k]
    #     # foreground features
    #     # foreground_feats["source"] = {"features": model.roi_heads.fg_features, "gt": model.roi_heads.iou_with_gt}
    #
    #     out_feats = {"backbone": backbone_feats, "foreground": foreground_feats}
    #     torch.save(out_feats, os.path.join(cfg.OUTPUT_DIR, "features_stat.pth"))
    if cfg.TEST.ANALYSIS:
        dataset_size = len(data_loader) * cfg.SOLVER.IMS_PER_BATCH_TEST
        # plot_pr(model.roi_heads.pr, os.path.join(cfg.OUTPUT_DIR, "PR_source.png"), "source",
        #         dataset_size)
        # plot_mh_dist(model.mh_distance, os.path.join(cfg.OUTPUT_DIR, "MH_distance_source.png".format(corrupt)), "source")
        # plot_pr_by_mh(model.roi_heads.pr_per_img, model.mh_distance,
        #               os.path.join(cfg.OUTPUT_DIR, "PR_by_MH_source.png"), "source")
        plot_tsne(model.roi_heads, cfg.OUTPUT_DIR)

    if len(results) == 1:
        results = list(results.values())[0]
    return results




def inference_on_dataset(
    cfg, model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    c_idx, wandb=None, evaluator_iter=None
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
    model.eval()
    model.requires_grad_(False)

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
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
            start_compute_time = time.perf_counter()
            outputs, _, _ = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

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

            cur_step = (c_idx * len(data_loader) + idx) * cfg.SOLVER.IMS_PER_BATCH_TEST
            # evaluate every iter
            if evaluator_iter is not None:
                evaluator_iter.reset()
                evaluator_iter.process(inputs, outputs)
                result_iter = evaluator_iter.evaluate(img_ids=[inp["image_id"] for inp in inputs])
                wandb.log({"EveryIter_AR@100": result_iter["box_proposals"]["AR@100"],
                           "EveryIter_AR@1000": result_iter["box_proposals"]["AR@1000"],
                           "EveryIter_mAP": result_iter["bbox"]["AP"],
                           "EveryIter_mAP50": result_iter["bbox"]["AP50"]}, step=cur_step)

    # plot_pr_stat(model.roi_heads.cnt, model.roi_heads.values, os.path.join(cfg.OUTPUT_DIR, "{}-top5.png".format(str(c_idx))))
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

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results
