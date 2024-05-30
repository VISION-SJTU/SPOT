import os
import glob
import torch
import traceback
from ltr.admin import multigpu
from ltr.utils.ext_loading import build_opt_lr_ocean
import ltr.utils.ltr_loading as loading


class BaseSparseTrainer:
    """
    Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function.
    """

    def __init__(self, actor, loader_assemble, optimizer, settings, middleware, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loader_assemble - Assembled dataset loaders
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            middleware - The middleware for connecting teacher and student networks
            lr_scheduler - Learning rate scheduler, typically used by SP-Ocean
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loader_assemble = loader_assemble
        self.middleware = middleware

        self.arch = self.actor.student_net.module.__class__.__name__
        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)

    def update_settings(self, settings=None):
        """
        Updates the trainer settings. Must be called to update internal settings.
        """
        if settings is not None:
            self.settings = settings

        if self.settings.env.var_dir is not None:
            self.settings.env.var_dir = os.path.expanduser(self.settings.env.var_dir)
            self._checkpoint_dir = os.path.join(self.settings.env.var_dir, 'checkpoints')
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True):
        """
        Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        epoch = -1
        num_tries = 10
        ending_tag = False
        for i in range(num_tries):
            if ending_tag:
                break
            try:
                # Load init checkpoint if necessary
                if load_latest:
                    self.load_checkpoint()

                # Cycle epochs
                for epoch in range(self.epoch + 1, max_epochs + 1):
                    self.epoch = epoch

                    # Main Loop for training
                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        try:
                            self.lr_scheduler.step()
                        except:
                            print("Now lr_scheduler reached an end.")
                            ending_tag = True

                    if self._checkpoint_dir and epoch % self.settings.ckp_saving_interval == 0:
                        self.save_checkpoint()

            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self, mode="all"):
        """
        Saves a checkpoint of the network and other variables.
        args:
            mode - Mode for whether to save teacher/student network, or both of them (all)
        """

        if mode == "all":
            self._save_student_checkpoint()
            self._save_teacher_checkpoint()
        elif mode == "teacher":
            self._save_teacher_checkpoint()
        elif mode == "student":
            self._save_student_checkpoint()
        else:
            assert False, "Saving checkpoint only supports mode all, student and teacher."

    def _save_teacher_checkpoint(self):
        """
        Saves a checkpoint of the teacher network and other variables.
        """

        teacher_net = self.actor.teacher_net.module if \
            multigpu.is_multi_gpu(self.actor.teacher_net) else self.actor.teacher_net

        # Only save necessary info and weight for teacher model
        net_type = type(teacher_net).__name__
        state_teacher = {
            'net_type': net_type,
            'net': teacher_net.state_dict(),
            'net_info': getattr(teacher_net, 'info', None),
            'constructor': getattr(teacher_net, 'constructor', None),
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path_teacher = '{}/{}_ep{:04d}_t.tmp'.format(directory, net_type, self.epoch)
        torch.save(state_teacher, tmp_file_path_teacher)

        file_path = '{}/{}_ep{:04d}_t.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path_teacher, file_path)

    def _save_student_checkpoint(self):
        """
        Saves a checkpoint of the student network and other variables.
        """

        student_net = self.actor.student_net.module if \
            multigpu.is_multi_gpu(self.actor.student_net) else self.actor.student_net

        actor_type = type(self.actor).__name__
        net_type = type(student_net).__name__
        state_student = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': student_net.state_dict(),
            'net_info': getattr(student_net, 'info', None),
            'constructor': getattr(student_net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path_student = '{}/{}_ep{:04d}_s.tmp'.format(directory, net_type, self.epoch)
        torch.save(state_student, tmp_file_path_student)

        file_path = '{}/{}_ep{:04d}_s.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path_student, file_path)

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False):
        """
        Loads a network checkpoint file for both teacher and student

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        student_net = self.actor.student_net.module if \
            multigpu.is_multi_gpu(self.actor.student_net) else self.actor.student_net
        teacher_net = self.actor.teacher_net.module if \
            multigpu.is_multi_gpu(self.actor.teacher_net) else self.actor.teacher_net

        actor_type = type(self.actor).__name__
        net_type = type(student_net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*_s.pth.tar'.format(self._checkpoint_dir,
                                                                               self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found.')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}_s.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                   net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # Checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*_s.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No matching checkpoint file found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load student network
        student_checkpoint_path = checkpoint_path
        student_checkpoint_dict = loading.torch_load_legacy(student_checkpoint_path)
        assert net_type == student_checkpoint_dict['net_type'], 'Network is not of correct type.'

        # Load teacher network
        teacher_checkpoint_path = checkpoint_path.replace("_s.pth.tar", "_t.pth.tar")
        teacher_checkpoint_dict = loading.torch_load_legacy(teacher_checkpoint_path)
        assert net_type == teacher_checkpoint_dict['net_type'], 'Network is not of correct type.'

        print("Loading existing checkpoint from {}".format(student_checkpoint_path.replace("_s.pth.tar", "")))

        # Now reload student network
        if fields is None:
            fields = student_checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

        # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Rebuild the optimizer and lr_scheduler for Ocean, if an existing checkpoint is loaded
        if self.arch in ["Ocean"]:
            self.optimizer, self.lr_scheduler = build_opt_lr_ocean(self.settings,
                                                                   student_net,
                                                                   student_checkpoint_dict['epoch'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                # Load weight for both teacher and student
                student_net.load_state_dict(student_checkpoint_dict[key])
                teacher_net.load_state_dict(teacher_checkpoint_dict[key])

            elif key == 'optimizer':
                # Only load optimizer from student
                self.optimizer.load_state_dict(student_checkpoint_dict[key])
            else:
                # Only load other attributes for training from student
                setattr(self, key, student_checkpoint_dict[key])

        # Reload constructor for both teacher and student
        if load_constructor:
            if 'constructor' in student_checkpoint_dict and \
                    student_checkpoint_dict['constructor'] is not None:
                student_net.constructor = student_checkpoint_dict['constructor']
            if 'constructor' in teacher_checkpoint_dict and \
                    teacher_checkpoint_dict['constructor'] is not None:
                teacher_net.constructor = teacher_checkpoint_dict['constructor']

        # Set the net info for both teacher and student
        if 'net_info' in student_checkpoint_dict and \
                student_checkpoint_dict['net_info'] is not None:
            student_net.info = student_checkpoint_dict['net_info']
        if 'net_info' in teacher_checkpoint_dict and \
                teacher_checkpoint_dict['net_info'] is not None:
            teacher_net.info = teacher_checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch

        return True
