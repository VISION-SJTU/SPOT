import datetime
from ltr.data import middleware, LTRSparseLoaderAssemble
import ltr.models.tracking.ocean_model as sp_models
import ltr.models.tracking.ocean_criterion as sp_criterions
import ltr.actor as actors
from ltr.trainers import LTRSparseTrainer
from ltr.admin.multigpu import MultiGPU
from ltr.dataset import Lasot, Got10k, TrackingNet
from ltr.utils.ext_loading import build_opt_lr_ocean, load_pretrain


def run(settings):
    # Common settings (e.g. dataset path) are assigned in config/local.py
    settings.device = 'cuda'
    settings.description = 'SPOT: Sparsely-Supervised Object Tracking. Version: SP-Ocean-3.'

    # Magic number, distinguishing training trials (used basically for recording sampler state)
    # now = datetime.datetime.now()
    # settings.magic_number = now.strftime("%Y-%m-%d-%H:%M:%S")
    settings.magic_number = "spot_ocean_003"

    # Options for dataloader
    # Batch sizes
    settings.batch_size_sup = 8
    settings.batch_size_unsup = 24
    settings.batch_size = settings.batch_size_sup + settings.batch_size_unsup
    # Num workers
    settings.num_workers_sup = 5
    settings.num_workers_unsup = 12
    settings.num_workers_burn_in = 16

    # Training settings
    # Multi-GPU tag
    settings.multi_gpu = True
    # Printing auxiliary info for training
    settings.printing_auxiliary_info = False
    # Iteration number every epoch train (T_iter in paper)
    settings.train_iteration_per_epoch = 1000
    # Train every N epochs (useless to some extent)
    settings.train_epoch_interval = 1

    # Network variants, turn off these tag only if you are conducting ablation study
    # Turn on this tag when using IoU filtering
    settings.iou_prediction_tag = True
    # Turn on this tag when using weak/strong augmentation
    settings.strong_augmentation_tag = True

    # Total epoch number
    settings.total_epochs = 350
    # Epoch number for burn-in stage
    settings.burn_in_epochs = 50
    # Optimizer parameters (following Ocean). We divide the epochs the make it compatible to our framework.
    settings.total_epoch_steps = 50
    settings.unfreeze_epoch_step = 10
    settings.opt_params = {
        'optimizers': [
            {'type': 'warmup', 'style': 'step', 'epoch_step': 5, 'start_lr': 0.001, 'end_lr': 0.005, 'step': 1},
            {'type': 'normal', 'style': 'log', 'epoch_step': 45, 'start_lr': 0.005, 'end_lr': 0.00001}],
        'base_lr': 0.005, 'momentum': 0.9, 'weight_decay': 0.0001, 'backbone_ratio': 0.1
    }
    settings.pretrain_backbone_path = './ltr/pretrain/imagenet_pretrain.model'
    settings.trainable_layers = ['layer1', 'layer2', 'layer3']
    assert settings.total_epochs % settings.total_epoch_steps == 0
    settings.epoch_each_lr_step = settings.total_epochs // settings.total_epoch_steps
    settings.unfreeze_epoch = settings.unfreeze_epoch_step * settings.epoch_each_lr_step

    # Parameter for EMA teacher network update (T_ema and \alpha in paper)
    settings.teacher_update_frequency = 5
    settings.keep_rate = 0.998

    # Epoch interval for printing essential information
    settings.print_interval = 1
    # Epoch interval for saving checkpoints
    settings.ckp_saving_interval = 25
    # Epoch interval for saving sampler state records (None for reserving only the latest state)
    settings.state_saving_interval = 100

    # Image transforms and essential parameters
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 31
    settings.template_feature_sz = 15
    settings.output_sz = 25
    settings.search_sz = 255
    settings.temp_sz = 127
    settings.stride = 8
    # Note: using weak augmentation for pseudo labeling, and strong augmentation for sparsely-supervised training
    settings.center_jitter_factor = {'search': 2.0, 'template': 0, 'unlabeled': 0, 'aug': 2.0}
    settings.scale_jitter_factor = {'search': 0.2, 'template': 0, 'unlabeled': 0, 'aug': 0.2}
    # Augmentation parameters
    settings.aug_settings = {'gray': 0.05, 'jitter': 0.2, 'angle': 10, 'lr_f_prob': 0.35, 'ud_f_prob': 0.1}

    # Ocean-related parameters
    settings.used_layers = [3]
    settings.before_neck_dim = 1024
    settings.head_dim = 256
    settings.head_tower_num = 4

    # Loss function parameters, for supervised and unsupervised instances
    settings.weight_dict = {'loss_ce_sup': 1, 'loss_bbox_sup': 1, 'loss_align_sup': 1, 'loss_iou_pred_sup': 0.5,
                            'loss_ce_unsup': 1, 'loss_bbox_unsup': 1, 'loss_align_unsup': 1}

    # Sparse-supervision settings (Comment, Uncomment and Revise Parameters for different settings)
    # For example, "key_frames" sampling mode: selecting and labeling only N (typically 3) key frames in each video
    settings.fs_settings = {'LaSOT': {'name': "LaSOT_train_kf_003",
                                      'split': "train", 'stage': ['burnin', 'sparse']},
                            'GOT-10k': {'name': "GOT-10k_vottrain_kf_003",
                                        'split': "vottrain", 'stage': ['burnin', 'sparse']},
                            'TrackingNet': {'name': "TrackingNet_0-3_kf_003",
                                            'split': "0-3", 'stage': ['burnin', 'sparse']}}

    # Settings for the training curriculum, including at least:
    #      "confidence_threshold": the confidence threshold for filtering tracking failure (\tau in paper)
    #      "max_gap": the maximum frame interval between template and search patch (used only for baseline)
    #      "propagating_step": the radium for propagating mode (r in paper)
    #      "window_penalty": the affecting ratio for temporal calibration
    settings.curriculum_settings = {'confidence_threshold': 0.82, 'max_gap': None,
                                    'propagating_step': 1, 'window_penalty': 0.30, 'cls_ratio': 0.94}

    # Print settings
    settings.print_settings()

    # Training datasets (using video datasets only)
    lasot_train = Lasot(split=settings.fs_settings['LaSOT']['split'])
    got10k_train = Got10k(split=settings.fs_settings['GOT-10k']['split'])
    trackingnet_train = TrackingNet(split=settings.fs_settings['TrackingNet']['split'])
    loader_datasets = {'train_datasets': [lasot_train, got10k_train, trackingnet_train], 'train_splits': [1, 1, 1]}

    # Assembled training loaders, see LTRSparseLoader for details
    loader_assemble = LTRSparseLoaderAssemble('loader_assemble', settings=settings, datasets=loader_datasets)

    # The middleware for processing and recording pseudo labels
    middleware_sp = middleware.SparseMiddleware(settings=settings)

    # Create teacher and student network
    teacher_model = sp_models.ocean_resnet50(settings, network="teacher")
    student_model = sp_models.ocean_resnet50(settings, network="student")
    # Load pretrained backbone manually for Ocean variant
    student_model = load_pretrain(student_model, settings.pretrain_backbone_path)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        teacher_model = MultiGPU(teacher_model, dim=0)
        student_model = MultiGPU(student_model, dim=0)

    # Loss function
    objective = sp_criterions.ocean_loss(settings)
    n_parameters = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Actor for teacher / student style sparsely-supervised training
    actor = actors.OceanSparseActor(settings=settings, student_net=student_model,
                                    teacher_net=teacher_model, objective=objective)

    # Optimizer and learning rate scheduler (following Ocean)
    optimizer, lr_scheduler = build_opt_lr_ocean(settings, student_model)

    # Create SPOT trainer
    trainer = LTRSparseTrainer(actor, loader_assemble, optimizer, settings, middleware_sp, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(settings.total_epochs, load_latest=True, fail_safe=False)
