import argparse
import os.path
from collections import OrderedDict
import torch
import pickle

"""
    backbone.conv1 -> backbone.bottom_up.stem.conv1.{weight}
    backbone.bn1 -> backbone.bottom_up.stem.norm
    backbone.layer1 -> backbone.bottom_up.res2
    backbone.layer1.0.downsample -> backbone.bottom_up.res2.0.shortcut
    backbone.layer1.0.downsample.0.weight -> backbone.bottom_up.res2.0.shortcut.weight
    backbone.layer1.0.downsample.1 -> backbone.bottom_up.res2.0.shortcut.norm.{weight, bias, running_mean, running_var}
    backbone.layer1.{0,1,2}.conv1.weight -> backbone.bottom_up.res2.{0,1,2}.conv1.weight
    backbone.layer1.{0,1,2}.bn1.{} -> backbone.bottom_up.res2.{0,1,2}.conv1.norm.{weight, bias, running_mean, running_var}
    backbone.layer1.{0,1,2}.conv2.weight -> backbone.bottom_up.res2.{0,1,2}.conv2.weight
    backbone.layer1.{0,1,2}.bn2.{} -> backbone.bottom_up.res2.{0,1,2}.conv2.norm.{weight, bias, running_mean, running_var}
    backbone.layer1.{0,1,2}.conv3.weight -> backbone.bottom_up.res2.{0,1,2}.conv3.weight
    backbone.layer1.{0,1,2}.bn3.{} -> backbone.bottom_up.res2.{0,1,2}.conv3.norm.{weight, bias, running_mean, running_var}
    neck -> backbone.fpn_*
        -> fpn_lateral{2,3,4,5}
        -> fpn_output{2,3,4,5}
    -> proposal_generator
        -> anchor_generator
        -> rpn_head
            -> conv
            -> objectness_logits
            -> anchor_deltas
    -> roi_heads
        -> box_head
            -> fc1
            -> fc2
        -> box_predictor
            -> cls_score
            -> box_predictor
"""

map_resnet_name = [
    ('backbone.conv1', 'backbone.bottom_up.stem.conv1'),
    ('backbone.bn1', 'backbone.bottom_up.stem.conv1.norm'),
    ('backbone.layer1', 'backbone.bottom_up.res2'),
    ('backbone.layer2', 'backbone.bottom_up.res3'),
    ('backbone.layer3', 'backbone.bottom_up.res4'),
    ('backbone.layer4', 'backbone.bottom_up.res5'),
    ('downsample.0', 'shortcut'),
    ('downsample.1', 'shortcut.norm'),
    ('bn1', 'conv1.norm'),
    ('bn2', 'conv2.norm'),
    ('bn3', 'conv3.norm'),
    ('neck.lateral_convs.0.conv', 'backbone.fpn_lateral2'),
    ('neck.lateral_convs.1.conv', 'backbone.fpn_lateral3'),
    ('neck.lateral_convs.2.conv', 'backbone.fpn_lateral4'),
    ('neck.lateral_convs.3.conv', 'backbone.fpn_lateral5'),
    ('neck.fpn_convs.0.conv', 'backbone.fpn_output2'),
    ('neck.fpn_convs.1.conv', 'backbone.fpn_output3'),
    ('neck.fpn_convs.2.conv', 'backbone.fpn_output4'),
    ('neck.fpn_convs.3.conv', 'backbone.fpn_output5'),
    ('rpn_head.rpn_conv', 'proposal_generator.rpn_head.conv'),
    ('rpn_head.rpn_cls', 'proposal_generator.rpn_head.objectness_logits'),
    ('rpn_head.rpn_reg', 'proposal_generator.rpn_head.anchor_deltas'),
    ('roi_head.bbox_head.shared_fcs.0', 'roi_heads.box_head.fc1'),
    ('roi_head.bbox_head.shared_fcs.1', 'roi_heads.box_head.fc2'),
    ('roi_head.bbox_head.fc_cls', 'roi_heads.box_predictor.cls_score'),
    ('roi_head.bbox_head.fc_reg', 'roi_heads.box_predictor.bbox_pred'),
]

def convert_resnet(src: str, dst: str) -> None:
    """Convert MMDetection checkpoint to Detectron2 style.

    Args:
        src (str): The MMDetection checkpoint path, should endswith `pkl`.
        dst (str): The Detectron2 checkpoint path.
        prefix (str): The prefix of MMDetection model, defaults to 'd2_model'.
    """
    "mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]"
    assert src.endswith('pkl') or src.endswith('pth'), 'the source Detectron2 checkpoint should endswith `pkl` or `pth`.'
    mm_model = torch.load(src)
    det2_model = pickle.load(open(dst, 'rb')) if dst.endswith('pkl') else torch.load(dst)
    det2_state_dict = det2_model['model']
    state_dict = {}
    print(det2_state_dict.keys())
    for name, value in mm_model['state_dict'].items():
        if 'num_batches_tracked' in name or 'teacher' in name:
            continue
        if not isinstance(value, torch.Tensor):
            value = torch.from_numpy(value)
        for m_key, d_key in map_resnet_name:
            name = name.replace('student.', '').replace(m_key, d_key)
        if value.shape != det2_state_dict[name].shape:
            print('MMdet Shape {} Detectron2 Shpae {} does not match!'.format(value.shape, det2_state_dict[name].shape))
        state_dict[name] = value

    ordered_state_dict = OrderedDict()
    for k in det2_state_dict.keys():
        if k not in list(state_dict.keys()):
            print('{} does not exists in MMdet!'.format(k))
            ordered_state_dict[k] = det2_state_dict[k]
        else:
            print('{}: {} in mmdet, {} in detectron2!'.format(k, torch.norm(state_dict[k]), torch.norm(torch.from_numpy(det2_state_dict[k]))))
            ordered_state_dict[k] = state_dict[k]

    torch.save(ordered_state_dict, './models/{}_student_converted_detectron2_style.pth'.format(os.path.basename(src).split('.')[0]))
    return


def convert_swintransformer(src: str, dst: str) -> None:
    """Convert MMDetection checkpoint to Detectron2 style.

    Args:
        src (str): The MMDetection checkpoint path, should endswith `pkl`.
        dst (str): The Detectron2 checkpoint path.
        prefix (str): The prefix of MMDetection model, defaults to 'd2_model'.
    """
    assert src.endswith('pkl') or src.endswith('pth'), 'the source Detectron2 checkpoint should endswith `pkl` or `pth`.'
    mm_model = torch.load(src)
    det2_model = pickle.load(open(dst, 'rb')) if dst.endswith('pkl') else torch.load(dst)
    det2_state_dict = det2_model['model']
    state_dict = {}
    print(det2_state_dict.keys())
    """
    mm style
    backbone
        -> patch_embed
            -> projection.{weight, bias}
            -> norm.{weight, bias}
        -> stages.{0~3}
            -> blocks.{0~}
                -> norm1
                -> attn
                    -> w_msa
                        -> relative_position_bias_table
                        -> relative_position_index
                        -> qkv.{weight, bias}
                        -> proj.{weight, bias}
                -> norm2.{weight, bias}
                -> ffn
                    -> layers.0.0.{weight, bias}
                    -> layers.1.{weight, bias}
            -> downsample
                -> norm.{weight, bias}
                -> reduction.weight
        -> norm{0~3}.{weight, bias}
    neck
        -> lateral_convs.{0~3}.conv.{weight, bias}
        -> fpn_convs.{0~3}.conv.{weight, bias}
    rpn_head
        -> rpn_conv.{weight, bias}
        -> rpn_cls.{weight, bias}
        -> rpn_reg.{weight, bias}
    roi_head
        -> bbox_head
            -> fc_cls.{weight, bias}
            -> fc_reg.{weight, bias}
            -> shared_fcs.{0~1}.{weight, bias}
    """
    """
    detectron2 style
    pixel_mean
    pixel_std
    backbone
        -> bottom_up
            -> patch_embed
                -> proj.{weight, bias}
                -> norm.{weight, bias}
            -> layers.{0~3}
            -> norm{0~3}
        -> fpn_lateral{2~5}
        -> fpn_output{2~5}
    proposal_generator
        -> rpn_head
            -> conv.{weight, bias}
            -> objectness_logits.{weight, bias}
            -> anchor_deltas.{weight, bias}
        -> anchor_generator.cell_anchors{0~4}
    roi_heads
        -> box_head
            -> fc{1~2}.{weight, bias}
        -> box_predictor
            -> cls_score.{weight, bias}
            -> bbox_pred.{weight, bias}
    """
    for name, value in mm_model['state_dict'].items():
        if not isinstance(value, torch.Tensor):
            value = torch.from_numpy(value)
        # convert backbone
        if 'backbone' in name:
            name = name.replace('backbone', 'backbone.bottom_up').replace('projection', 'proj').replace('stages', 'layers')
            name = name.replace('attn.w_msa', 'attn')
            name = name.replace('ffn.layers.0.0', 'mlp.fc1')
            name = name.replace('ffn.layers.1', 'mlp.fc2')
        # convert neck
        if 'neck' in name:
            for n in range(4):
                name = name.replace('neck.lateral_convs.{}.conv'.format(str(n)), 'backbone.fpn_lateral{}'.format(str(n + 2)))
                name = name.replace('neck.fpn_convs.{}.conv'.format(str(n)), 'backbone.fpn_output{}'.format(str(n + 2)))
        # convert rpn head
        if 'rpn_head' in name:
            name = name.replace('rpn_head.rpn_conv', 'proposal_generator.rpn_head.conv')
            name = name.replace('rpn_head.rpn_cls', 'proposal_generator.rpn_head.objectness_logits')
            name = name.replace('rpn_head.rpn_reg', 'proposal_generator.rpn_head.anchor_deltas')
        # convert roi head
        if 'roi_head' in name:
            name = name.replace('roi_head.bbox_head.shared_fcs.0', 'roi_heads.box_head.fc1')
            name = name.replace('roi_head.bbox_head.shared_fcs.1', 'roi_heads.box_head.fc2')
            name = name.replace('roi_head.bbox_head.fc_cls', 'roi_heads.box_predictor.cls_score')
            name = name.replace('roi_head.bbox_head.fc_reg', 'roi_heads.box_predictor.bbox_pred')
        if value.shape != det2_state_dict[name].shape:
            print('{} MMdet Shape {} Detectron2 Shpae {} does not match!'.format(name, value.shape, det2_state_dict[name].shape))
        state_dict[name] = value
    state_dict['pixel_mean'] = torch.Tensor([123.675, 116.28, 103.53])
    state_dict['pixel_std'] = torch.Tensor([58.395, 57.12, 57.375])
    ordered_state_dict = OrderedDict()
    for k in det2_state_dict.keys():
        if k not in list(state_dict.keys()):
            print('{} does not exists in MMdet!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(k))
            ordered_state_dict[k] = det2_state_dict[k]
        else:
            try:
                print('{}: {} in mmdet, {} in detectron2!'.format(k, torch.norm(state_dict[k]), torch.norm(det2_state_dict[k])))
            except:
                print('{}: {} in mmdet, {} in detectron2!'.format(k, state_dict[k], det2_state_dict[k]))
            ordered_state_dict[k] = state_dict[k]

    torch.save(ordered_state_dict, './models/{}_converted_detectron2_style.pth'.format(os.path.basename(src).split('.')[0]))
    return


def main():
    parser = argparse.ArgumentParser(
        description='Convert Detectron2 checkpoint to MMDetection style')
    parser.add_argument('--src', default='./models/stfar_coco_r50_fpn.pth', help='Detectron2 model path')
    parser.add_argument('--dst', default='./models/source_coco_r50_fpn_3x.pkl', help='reference model path')
    parser.add_argument('--save_path', default=None, help='Converted state dict save path')
    parser.add_argument('--arch', default='resnet', help='model type. renset or swintransformer')

    args = parser.parse_args()
    if args.arch == 'resnet':
        convert_resnet(args.src, args.dst)
    else:
        convert_swintransformer(args.src, args.dst)

if __name__ == '__main__':
    main()