from abc import ABCMeta, abstractmethod


class CubeTaskModule(metaclass=ABCMeta):
    @abstractmethod
    def load_checkpoint(self, ckpt_path: str, *args, **kwargs) -> "CubeTaskModule":
        pass


class CubeDataModule(metaclass=ABCMeta):
    pass


class CubeRunner(metaclass=ABCMeta):
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def validate(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
