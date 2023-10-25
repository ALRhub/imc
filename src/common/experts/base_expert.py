from abc import ABC, abstractmethod


class BaseExpert(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def log_likelihood(self, obs, act):
        pass

    @abstractmethod
    def sample(self, obs, cmp_idx=None):
        pass
