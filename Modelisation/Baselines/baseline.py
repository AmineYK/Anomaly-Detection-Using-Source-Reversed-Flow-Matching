from abc import ABC, abstractmethod

class BaselineModel(ABC):

    def __init__(self, *args):
        self.args = args

    @abstractmethod
    def train(self, X_train):
        pass
    
    @abstractmethod
    def test(self, X_test, y_test=None):
        pass
