from ltr.utils import TensorDict


class BaseSparseActor:
    """
        Base class for sparsely-supervised tracking actor.
        The actor class handles the passing of the data through the network
        and calculation the loss
    """
    def __init__(self, settings, student_net, teacher_net, objective):
        """
        args:
            student_net - The student network to train
            student_net - The teacher network for pseudo labeling
            objective - The loss function
            settings - Important training settings
        """
        self.student_net = student_net
        self.teacher_net = teacher_net
        self.objective = objective
        self.settings = settings

    def __call__(self, data: TensorDict):
        """
        Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            data - A TensorDict containing all the necessary data blocks.
            Other inputs varies according to the calling mode, please refer to detailed implementation classes
        """
        raise NotImplementedError

    def to(self, device):
        """
        Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.student_net.to(device)
        self.teacher_net.to(device)

    def train(self, mode=True):
        """
        Set whether the student network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.student_net.train(mode)
        self.objective.train(mode)

    def eval(self):
        """ Set network to eval mode """
        self.train(False)
