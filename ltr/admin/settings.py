from ltr.admin.environment import env_settings


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()

    def set_default(self):
        self.env = env_settings()
        self.use_gpu = True

    def print_settings(self):
        print("\nEnvironment settings: \n" +
              '\n'.join(['%s: %s' % item for item in self.env.__dict__.items() if item[1] != ""]))
        print("\nTraining settings: \n" +
              '\n'.join(['%s: %s' % item for item in self.__dict__.items() if item[0] != 'env']) + '\n')
