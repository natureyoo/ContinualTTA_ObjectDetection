from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.fcos import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
dataloader.test.batch_size = 4
optimizer.lr = 0.01

model.backbone.bottom_up.freeze_at = 2
train.init_checkpoint = "outputs/fcos_r50/model_final.pth"
train.output_dir = "outputs/fcos_r50_adapt"

model.collect_features = False
model.online_adapt = True
model.gl_align = 'KL'
model.fg_align = None
model.source_feat_stats = 'models/fcos_feature_stats.pt'
model.ema_gamma = 128

# where
model.backbone.bottom_up.stages.adapter = 'parallel' # None
