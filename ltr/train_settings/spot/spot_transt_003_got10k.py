import torch, datetime
from ltr.data import middleware, LTRSparseLoaderAssemble
import ltr.models.tracking.transt_model as sp_models
import ltr.models.tracking.transt_criterion as sp_criterions
import ltr.actor as actors
from ltr.trainers import LTRSparseTrainer
from ltr.admin.multigpu import MultiGPU
from ltr.dataset import Lasot, Got10k, TrackingNet


def run(settings):
    # Common settings (e.g. dataset path) are assigned in config/local.py
    settings.device = 'cuda'
    settings.description = 'SPOT: Sparsely-Supervised Object Tracking. Version: SP-TransT-3-GOT.'

    # Magic number, distinguishing training trials (used basically for recording sampler state)
    # now = datetime.datetime.now()
    # settings.magic_number = now.strftime("%Y-%m-%d-%H:%M:%S")
    settings.magic_number = "spot_transt_003_got10k"

    # Options for dataloader
    # Batch sizes
    settings.batch_size_sup = 8
    settings.batch_size_unsup = 24
    settings.batch_size = settings.batch_size_sup + settings.batch_size_unsup
    # Num workers
    settings.num_workers_sup = 5
    settings.num_workers_unsup = 12
    settings.num_workers_burn_in = 10

    # Training settings
    # Multi-GPU tag
    settings.multi_gpu = True
    # Printing Auxiliary info for training
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
    # Epoch for optimizer step (following TransT)
    settings.optimizer_step_epochs = 250
    # Optimizer parameters (following TransT)
    settings.optimizer_parameters = {'learning_rate': 1e-4, 'weight_decay': 1e-4, 'backbone_ratio': 0.1}

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
    settings.search_feature_sz = 32
    settings.template_feature_sz = 16
    settings.search_sz = settings.search_feature_sz * 8
    settings.temp_sz = settings.template_feature_sz * 8
    # Note: using weak augmentation for pseudo labeling, and strong augmentation for sparsely-supervised training
    settings.center_jitter_factor = {'search': 3, 'template': 0.0, 'unlabeled': 0.0, 'aug': 3}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0.0, 'unlabeled': 0.0, 'aug': 0.25}
    # Augmentation parameters
    settings.aug_settings = {'gray': 0.05, 'jitter': 0.2, 'angle': 10, 'lr_f_prob': 0.35, 'ud_f_prob': 0.10}

    # Transformer-related parameters
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 4

    # Loss function parameters, for supervised and unsupervised instances
    settings.num_classes = 1
    settings.eos_coef = 0.0625
    settings.weight_dict = {'loss_ce_sup': 8.334, 'loss_bbox_sup': 5, 'loss_giou_sup': 2, 'loss_iou_pred_sup': 1.0,
                            'loss_ce_unsup': 8.334, 'loss_bbox_unsup': 5, 'loss_giou_unsup': 2}

    # Sparse-supervision settings (Comment, Uncomment and Revise Parameters for different settings)
    # For example, "key_frames" sampling mode: selecting and labeling only N (typically 3) key frames in each video
    settings.fs_settings = {'GOT-10k': {'name': "GOT-10k_all_kf_003",
                                        'split': "all", 'stage': ['burnin', 'sparse']}}

    # Settings for the training curriculum, including at least:
    #      "confidence_threshold": the confidence threshold for filtering tracking failure (\tau in paper)
    #      "max_gap": the maximum frame interval between template and search patch (used only for baseline)
    #      "propagating_step": the radium for propagating mode (r in paper)
    #      "window_penalty": the affecting ratio for temporal calibration
    settings.curriculum_settings = {'confidence_threshold': 0.85, 'max_gap': None,
                                    'propagating_step': 1, 'window_penalty': 0.40}

    # Print settings
    settings.print_settings()

    # Training datasets (using video datasets only)
    got10k_train = Got10k(split=settings.fs_settings['GOT-10k']['split'])
    loader_datasets = {'train_datasets': [got10k_train], 'train_splits': [1]}

    # Assembled training loaders, see LTRSparseLoaderAssemble for details
    loader_assemble = LTRSparseLoaderAssemble('loader_assemble', settings=settings, datasets=loader_datasets)

    # The middleware for processing and recording pseudo labels
    middleware_sp = middleware.SparseMiddleware(settings=settings)

    # Create teacher and student network
    teacher_model = sp_models.transt_resnet50(settings, network="teacher")
    student_model = sp_models.transt_resnet50(settings, network="student")

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        teacher_model = MultiGPU(teacher_model, dim=0)
        student_model = MultiGPU(student_model, dim=0)

    # Loss function
    objective = sp_criterions.transt_loss(settings)
    n_parameters = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Actor for teacher / student style sparsely-supervised training
    actor = actors.TranstSparseActor(settings=settings, student_net=student_model,
                                     teacher_net=teacher_model, objective=objective)

    # Optimizer (following TransT)
    param_dicts = [
        {"params": [p for n, p in student_model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in student_model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": settings.optimizer_parameters['backbone_ratio'] * settings.optimizer_parameters['learning_rate'],
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=settings.optimizer_parameters['learning_rate'],
                                  weight_decay=settings.optimizer_parameters['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, settings.optimizer_step_epochs)

    # Create SPOT trainer
    trainer = LTRSparseTrainer(actor, loader_assemble, optimizer, settings, middleware_sp, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(settings.total_epochs, load_latest=True, fail_safe=False)
