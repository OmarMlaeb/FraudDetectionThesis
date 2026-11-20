import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
except ImportError as exc:
    raise ImportError(
        "PyTorch Geometric is required for the GNN baselines. "
        "Install it with: pip install torch-geometric"
    ) from exc


class GCNFraudDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(GCNFraudDetector, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(1)


class GraphSAGEFraudDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(GraphSAGEFraudDetector, self).__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(1)


class GATFraudDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, heads=2, dropout=0.3):
        super(GATFraudDetector, self).__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(1)


def create_gnn_model(model_name, input_dim):
    model_name = model_name.lower()

    if model_name == "gcn":
        return GCNFraudDetector(input_dim)
    if model_name == "sage":
        return GraphSAGEFraudDetector(input_dim)
    if model_name == "gat":
        return GATFraudDetector(input_dim)

    raise ValueError("model_name must be one of: gcn, sage, gat")
