class Probabilistic:
    def __init__(self, episode_size, distribution):
        self.distribution = distribution
        self.episode_size = episode_size
        self.__saved = None

    def _to_backward(self, value):
        self.__saved = value

    def _from_forward(self):
        value = self.__saved
        self.__saved = None
        return value

    def sample(self, params, shape=None):
        dist = self.distribution(*params)
        return dist.sample() if shape is None else dist.sample(shape)

    def grad_samples(self, params):
        raise NotImplementedError

    def backward(self, losses):
        raise NotImplementedError
