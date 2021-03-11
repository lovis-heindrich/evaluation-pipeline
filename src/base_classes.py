from abc import ABC, abstractmethod 

class PipelineClassifier(ABC):
    @abstractmethod
    def get_prediction(self, x):
        pass

    @abstractmethod
    def get_name(self):
        pass