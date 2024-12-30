from abc import ABC, abstractmethod 

class ExecutorBase(ABC):
    @abstractmethod 
    def execute_model(
        self, req
    ):
        pass
