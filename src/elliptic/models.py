import torch
import torch.nn as nn


class MLPFraudDetector(nn.Module):
    def __init__(self, input_dim):
        super(MLPFraudDetector, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


class LSTMFraudDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.3):
        super(LSTMFraudDetector, self).__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_timestep = output[:, -1, :]
        return self.classifier(last_timestep).squeeze(1)


class TransformerFraudDetector(nn.Module):
    def __init__(
        self,
        input_dim,
        sequence_length,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.3,
    ):
        super(TransformerFraudDetector, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_projection(x) + self.position_embedding[:, : x.size(1), :]
        encoded = self.encoder(x)
        last_timestep = encoded[:, -1, :]
        return self.classifier(last_timestep).squeeze(1)
