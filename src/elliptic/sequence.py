import torch

from torch.utils.data import Dataset


class EllipticSequenceDataset(Dataset):
    def __init__(self, X, y, mask, sequence_length):
        labeled_indices = torch.where(torch.tensor(mask, dtype=torch.bool))[0]
        if len(labeled_indices) == 0:
            raise ValueError("Split has no labeled examples.")

        self.X = torch.tensor(X[labeled_indices])
        self.y = torch.tensor(y[labeled_indices])
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        start = max(0, index - self.sequence_length + 1)
        window = self.X[start:index + 1]

        if len(window) < self.sequence_length:
            padding = window[0:1].repeat(self.sequence_length - len(window), 1)
            window = torch.cat((padding, window), dim=0)

        return window, self.y[index]


def predict_sequence_model(model, data_loader, device):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits).cpu()

            all_probs.append(probs)
            all_targets.append(batch_y)

    return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()
