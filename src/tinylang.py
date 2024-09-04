import torch
from torch import nn, optim

from utils import tokenize_words, tokenize_chars, compute_vocabulary

class NextTokenPredictionModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int) -> None:
        super().__init__()
        self.vocab_size =  vocab_size
        self.context_length = context_length
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class NextTokenPrediction:
    def __init__(self, vocabulary: dict[str, int], context_length: int, max_iterations: int=1) -> None:
        self.model = NextTokenPrediction(len(vocabulary), context_length)
        self.optimizer = optim.Adam()
        self.loss = nn.CrossEntropyLoss()
        self.max_iterations = max_iterations

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass