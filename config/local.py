import os


class EnvironmentSettings:
    def __init__(self):
        # Base directory for project workspace
        self.workspace_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        # Var directory for project, storing everything changing for deployment
        self.var_dir = os.path.join(self.workspace_dir, "var")
        # Tensorboard directory for project
        self.tensorboard_dir = os.path.join(self.var_dir, "tensorboard")

        # Directories for training datasets
        self.train_dataset_dirs = {
            'LaSOT': '/ssd_4t/jlzheng/LaSOT/LaSOTBenchmark',
            'GOT-10k': '/ssd_4t/jlzheng/GOT-10k/train',
            'TrackingNet': '/ssd_2t/TrackingNet',
        }

        # Directories for testing datasets
        self.test_dataset_dirs = {
            'LaSOT': '/ssd_4t/jlzheng/LaSOT/LaSOTTesting',
            'GOT-10k': '/ssd_4t/jlzheng/GOT-10k/test',
            'TrackingNet': '/ssd_2t/TrackingNet/TEST/frames',
            'OTB100': '/data/jlzheng/test/OTB100',
            'NFS': '/home/jlzheng/dataset/tracking/Nfs',
            'VOT2016': '/home/jlzheng/dataset/tracking/VOT2016',
            'VOT2018': '/home/jlzheng/dataset/tracking/VOT2018',
            'UAV123': '/home/jlzheng/dataset/tracking/UAV123',
        }
