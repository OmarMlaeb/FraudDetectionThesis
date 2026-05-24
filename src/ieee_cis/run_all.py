from .train_gnn import train as train_gnn
from .train_lstm import train as train_lstm
from .train_mlp import train as train_mlp
from .train_transformer import train as train_transformer


def run_all():
    train_mlp()
    train_lstm()
    train_transformer()
    train_gnn(model_name="gcn")
    train_gnn(model_name="sage")
    train_gnn(model_name="gat")


if __name__ == "__main__":
    run_all()
