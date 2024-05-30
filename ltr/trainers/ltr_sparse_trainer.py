import os
from collections import OrderedDict
from ltr.trainers import BaseSparseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter
from ltr.utils import TensorDict
import torch
import time
from ltr.utils.ext_loading import build_opt_lr_ocean


class LTRSparseTrainer(BaseSparseTrainer):
    def __init__(self, actor, loader_assemble, optimizer, settings, middleware, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loader_assemble - Assembled dataset loaders
            optimizer - The optimizer used for training, e.g. AdamW
            settings - Training settings
            middleware - The middleware for connecting teacher and student networks
            lr_scheduler - Learning rate scheduler, typically used by SP-Ocean
        """

        super().__init__(actor, loader_assemble, optimizer, settings, middleware, lr_scheduler)

        self.backbone_frozen = True if self.arch in ["Ocean"] else False
        self._set_default_settings()

        # Initialize statistics variables
        loader_names = ['train']
        self.stats = OrderedDict({loader_name: None for loader_name in loader_names})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, loader_names)

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset_burnin(self, loader):
        """
        Do a cycle of training in burn-in stage.
        """

        # Switch student model to training mode
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        # Time init
        self._init_timing()

        for i, data in enumerate(loader, 1):

            # Forward pass
            loss, stats = self._burnin_training(data)

            # Skip this iteration if something unexpected happened in student net
            if loss is None:
                print("Skip an iteration!")
                continue

            # Backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clip for Ocean
                if self.arch in ['Ocean']:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(self.actor.student_net.module.parameters(), 25)
                    if gradient_norm.isnan() or gradient_norm.isinf():
                        print("Gradient is nan/inf, loss is {},so skipping this iteration!".format(loss.item()))
                        continue
                self.optimizer.step()

            # Update statistics
            batch_size = self.settings.batch_size
            self._update_stats(stats, batch_size, loader)

            # Print statistics
            self._print_stats(i, loader, batch_size)

    def cycle_dataset_sparse_supervised(self, loader_sup, loader_unsup):
        """
        Do a cycle of training in sparsely-supervised training stage.
        """

        # Switch student model to training mode
        self.actor.train(loader_unsup.training)
        torch.set_grad_enabled(loader_unsup.training)

        # Time init
        self._init_timing()

        for i, (data_sup, (data_unsup, middle_info)) in enumerate(zip(loader_sup, loader_unsup)):

            # Update teacher model periodically
            if (i + 1) % self.settings.teacher_update_frequency == 0:
                self._teacher_model_update(keep_rate=self.settings.keep_rate)

            # Teacher network inference
            pseudo_label = self._pseudo_label_generation(data_unsup)

            # Record inference results in sampler state, and refresh the state
            aug_label = self._middleware_ops(middle_info, pseudo_label)

            # Train student model with pseudo labels
            loss, stats = self._sparse_training(data_sup, data_unsup, aug_label)

            # Skip this iteration if something unexpected happened in student net
            if loss is None:
                print("Skip an iteration!")
                continue

            # Backward pass and update weights
            if loader_unsup.training:
                self.optimizer.zero_grad()
                loss.backward()
                if self.arch in ['Ocean']:
                    # Ocean may encounter nan gradient with a very small prob, which will cause everything collapse
                    gradient_norm = torch.nn.utils.clip_grad_norm_(self.actor.student_net.module.parameters(), 25)
                    if gradient_norm.isnan() or gradient_norm.isinf():
                        print("Gradient is nan/inf, so skipping this iteration!")
                        continue
                self.optimizer.step()

            # Update statistics
            self._update_stats(stats, self.settings.batch_size, loader_unsup)
            # Print statistics
            self._print_stats(i, loader_unsup, self.settings.batch_size)

        # Update the sampler state at the end of each epoch
        self.middleware.update_sampler_state(self.epoch)

    def train_epoch(self):
        """
        Do one epoch for each loader.
        """

        if self.arch in ["Ocean"]:
            # Unfreeze backbone for Ocean
            if self.epoch == self.settings.unfreeze_epoch:
                optimizer, lr_scheduler = build_opt_lr_ocean(self.settings,
                                                             self.actor.student_net.module,
                                                             self.epoch - 1)
                self.optimizer = optimizer
                self.lr_scheduler = lr_scheduler
                self.backbone_frozen = False

        if self.epoch <= self.settings.burn_in_epochs:

            # Burn-in stage, only use sparsely sampled labeled data
            # Init the burn-in loader at the beginning of each epoch
            if self.epoch % self.settings.train_epoch_interval == 0:
                self.loader_assemble.init_burnin_loader(self.epoch)
                self.cycle_dataset_burnin(self.loader_assemble.loader_burnin)

        else:
            # End of burn-in stage, copy burn-in weight from student to teacher
            if self.epoch == self.settings.burn_in_epochs + 1:
                self._teacher_model_update(keep_rate=0)

            # Sparsely-supervised stage, use teacher-student style to train
            if self.epoch % self.settings.train_epoch_interval == 0:
                # Init the sparsely-supervised loader at the beginning of each epoch
                self.loader_assemble.init_sparse_supervised_loader(self.epoch)
                self.cycle_dataset_sparse_supervised(self.loader_assemble.loader_sup,
                                                     self.loader_assemble.loader_unsup)

        # Emptying GPU cache at the end of each epoch
        torch.cuda.empty_cache()
        # Record necessary info
        self._stats_new_epoch()
        self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        """
        Initialize stats if not initialized yet.
        """

        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):

        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        """
        Record learning rate
        """

        lr_list = self.lr_scheduler.get_lr()
        for i, lr in enumerate(lr_list):
            var_name = 'LearningRate/group{}'.format(i)
            if var_name not in self.stats['train'].keys():
                self.stats['train'][var_name] = StatValue()
            self.stats['train'][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name,
                                               self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

    def _teacher_model_update(self, keep_rate=0.998):
        """
        EMA update the teacher network with student network
        Detailed implementation is found in SparseActor, mode "update"
        """

        # Switch student model to inference mode
        self.actor.train(False)
        torch.set_grad_enabled(False)

        # Update weight of teacher model from student model
        self.actor(mode="update", keep_rate=keep_rate)

    def _burnin_training(self, data):
        """
        Training the student network in a burn-in style
        Detailed implementation is found in SparseActor, mode "burnin"
        """

        if self.move_data_to_gpu:
            data = data.to(self.device)

        # Transform and joint augmentation (weak aug) are needed for burn-in, other than strong augmentation.
        aug_mask = torch.zeros(self.settings.batch_size, dtype=torch.bool, device=self.device)
        data['masks'] = aug_mask

        # Joint augmentation (weak aug) is needed for burn-in, other than strong aug
        loss, stats = self.actor(mode="burnin", burnin_data=data)
        # Extreme case
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            print("Loss becomes nan/inf during training, so we will skip this iteration.")
            return None, None

        return loss, stats

    def _middleware_ops(self, middle_info, pseudo_label):
        """
        This function assembles two things:
            1. Re-crop data with pseudo bbox,
                and generate corresponding sparsely-supervised instances for student net
            2. Record the sampler state on unlabeled frames in the MiddleWare for each iteration,
                        which will be updated on disk at the end of each epoch (for subsequent epochs).
        """

        return self.middleware.record_relay_results(middle_info, pseudo_label)

    def _pseudo_label_generation(self, data):
        """
        Inference through the teacher network to obtain the pseudo labels on unlabeled frames
        Detailed implementation is found in SparseActor, mode "inference"
        """

        # Switch student model to inference mode
        self.actor.train(False)
        torch.set_grad_enabled(False)

        # Load data for pseudo label generation phase
        # Labeled image (key image) as template, to generate pseudo labels for unlabeled search areas
        pres, sufs = ['key', 'unlabeled'], ['images', 'anno', 'avg']
        data_load = TensorDict()
        for prefix in pres:
            for suffix in sufs:
                data_load['{}_{}'.format(prefix, suffix)] = data['{}_{}'.format(prefix, suffix)]
        if self.move_data_to_gpu:
            data_load = data_load.to(self.device)

        # Tell the network that all instances do not need strong augmentation
        data_load['masks'] = torch.zeros(self.settings.batch_size_unsup, dtype=torch.bool, device=self.device)

        # forward pass for generating pseudo label
        pseudo_label = self.actor(mode="inference", teacher_data=data_load)
        return pseudo_label

    def _sparse_training(self, data_sup, data_unsup, aug_label, training=True):
        """
        Training the student network in sparsely-supervised style
        Detailed implementation is found in SparseActor, mode "spsup"
        """

        # Switch student model to inference mode
        self.actor.train(training)
        torch.set_grad_enabled(training)
        # Stack labeled instances and pseudo instances
        batch_size_sup = self.settings.batch_size_sup
        batch_size_unsup = self.settings.batch_size_unsup

        # Extreme case: An iteration with all failed tracking attempts
        batch_unsup_already_sampled = sum((aug_label['aug_success'])).item()
        if batch_unsup_already_sampled == 0 and batch_size_unsup > 0:
            print("Found no valid pseudo instance generated by teacher in this batch. skip iteration!")
            return None, None

        # Labeled image (key image) as template, to generate pseudo labels for unlabeled search areas
        # Fill in the augmented label in TensorDict
        data_unsup['aug_anno'] = aug_label['aug_anno']
        pres, sufs = ['relay', 'aug'], ['images', 'anno', 'avg']
        data_load = TensorDict()
        for prefix in pres:
            for suffix in sufs:
                key = '{}_{}'.format(prefix, suffix)
                gather = []
                for b in range(batch_size_unsup):
                    # Fill in only successful tracking attempts
                    if aug_label['aug_success'][b]:
                        gather.append(data_unsup[key][b])
                data_load[key] = torch.stack(gather, dim=0)

        # Some unreliable instances are removed above for low confidence score
        # We should clone reliable instances to ensure a stable batch size (preventing unstable GPU consumption)
        for key in data_load:
            volatile_bas = batch_unsup_already_sampled
            while volatile_bas < batch_size_unsup:
                num_to_fill = min(batch_size_unsup - volatile_bas, volatile_bas)
                inst_to_fill = data_load[key][0:num_to_fill]
                data_load[key] = torch.cat([data_load[key], inst_to_fill])
                volatile_bas += num_to_fill
        # Refresh data_unsup as data_load after instance filtering and filling
        data_unsup = data_load

        # Move data to GPU
        if self.move_data_to_gpu:
            data_sup = data_sup.to(self.device)
            data_unsup = data_unsup.to(self.device)

        # Labeled masks indicate whether the instances are supervised or unsupervised pairs
        data_sup['masks'] = torch.zeros(batch_size_sup, dtype=torch.bool, device=self.device)
        data_unsup['masks'] = torch.ones(batch_size_unsup, dtype=torch.bool, device=self.device)

        # Forward pass
        loss, stats = self.actor(mode="spsup", burnin_data=data_sup, sp_data=data_unsup)

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            print("Loss becomes nan/inf during training, will skip this iteration.")
            return None, None

        return loss, stats

