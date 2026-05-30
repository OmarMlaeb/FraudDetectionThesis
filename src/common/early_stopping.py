import torch


class EarlyStopping:
    def __init__(self, patience, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def step(self, score, model, model_path, epoch):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience
