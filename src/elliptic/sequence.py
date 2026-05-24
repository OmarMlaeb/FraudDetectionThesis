import torch

from torch.utils.data import Dataset


class EllipticSequenceDataset(Dataset):
    def __init__(self, X, y, mask, sequence_length):
        labeled_indices = torch.where(torch.tensor(mask, dtype=torch.bool))[0]
        if len(labeled_indices) < sequence_length:
            raise ValueError("Split is smaller than the requested sequence length.")

        self.X = torch.tensor(X[labeled_indices])
        self.y = torch.tensor(y[labeled_indices])
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, index):
        end = index + self.sequence_length
        return self.X[index:end], self.y[end - 1]


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
