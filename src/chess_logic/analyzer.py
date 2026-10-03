from abc import ABC, abstractmethod


class PositionAnalyzer(ABC):

    @abstractmethod
    def analyze(self, board):
        pass