"""Memory-efficient GCN/SAGE aggregation; frozen model parameters stay unchanged."""
import types
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, remove_self_loops

BACKEND = 'native_csr_gnn_v1'


class EdgeLayout:
    """CSR row layouts in both directions, retaining the original edge order."""
    def __init__(self, edges, nodes):
        self.edges, self.nodes = edges, nodes
        self.layouts = []
        for row, col in [edges.flip(0), edges]:
            order = torch.argsort(row * nodes + col)
            counts = torch.bincount(row, minlength=nodes)
            pointer = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
            self.layouts.append((pointer, col[order], order))

    def multiply(self, values, x, transpose=False):
        pointer, columns, order = self.layouts[int(transpose)]
        adjacency = torch.sparse_csr_tensor(pointer, columns, values[order],
                                             size=(self.nodes, self.nodes))
        return torch.sparse.mm(adjacency, x.contiguous())


class AttentionSum(torch.autograd.Function):
    """Exact weighted neighbor sum, without an edges-by-heads-by-features buffer.

    Sparse value gradients are evaluated in bounded chunks. This avoids both
    dense N-by-N gradients and giant gathered edge-feature tensors. The algebra
    is the original GAT message sum; only the reduction order changes.
    """
    @staticmethod
    def forward(ctx, x, alpha, layout):
        ctx.save_for_backward(x, alpha)
        ctx.layout = layout
        return torch.stack([layout.multiply(alpha[:, h], x[:, h])
                            for h in range(x.shape[1])], dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        layout = ctx.layout
        grad_x = torch.stack([layout.multiply(alpha[:, h], grad_output[:, h], transpose=True)
                              for h in range(x.shape[1])], dim=1)
        grad_alpha = torch.empty_like(alpha)
        # Same derivative as sum(alpha[e,h] * x[source[e],h,c]).
        for start in range(0, alpha.shape[0], 65536):
            stop = start + 65536
            src, dst = layout.edges[:, start:stop]
            grad_alpha[start:stop] = (x[src] * grad_output[dst]).sum(dim=-1)
        return grad_x, grad_alpha, None


def accelerate(model, name):
    """Use the same full graph, weights, normalization, activations and dropout.

    Non-graph models keep their original implementation. Sparse sums can
    differ in floating-point rounding; this is recorded as a different backend.
    The cached adjacency is deliberately not a checkpoint parameter/buffer.
    """
    if name not in ('gcn', 'sage', 'gat'):
        return model
    original_forward = model.forward
    cache = {}

    if name == 'gat':
        for conv in [model.conv1, model.conv2]:
            if conv.lin is None or conv.res is not None or conv.edge_dim is not None:
                raise ValueError('Sparse GAT is restricted to the frozen homogeneous models.')
            conv.add_self_loops = False

            def propagate(self, edge_index, x, alpha, size=None):
                # Original GAT.forward and edge_updater still compute attention,
                # softmax and dropout in the original edge order.
                return AttentionSum.apply(x[0], alpha, cache['layout'])

            conv.propagate = types.MethodType(propagate, conv)

    def forward(self, x, edge_index):
        key = (edge_index.data_ptr(), edge_index._version, tuple(edge_index.shape),
               x.shape[0], x.device, x.dtype)
        if cache.get('key') != key:
            edges = edge_index
            if name == 'gat':
                edges, _ = remove_self_loops(edges)
                edges, _ = add_self_loops(edges, num_nodes=x.shape[0])
                cache.update(key=key, adjacency=edges, layout=EdgeLayout(edges, x.shape[0]))
                return original_forward(x, edges)
            values = torch.ones(edges.shape[1], device=x.device, dtype=x.dtype)
            if name == 'gcn':
                # Normalize in the original edge order, including the original
                # self-loop convention, BEFORE sorting into sparse CSR.
                conv = self.conv1
                edges, values = gcn_norm(edges, None, x.shape[0], conv.improved,
                                         conv.add_self_loops, conv.flow, x.dtype)
            adjacency = torch.sparse_coo_tensor(edges.flip(0), values,
                                                (x.shape[0], x.shape[0])).coalesce().to_sparse_csr()
            cache.update(key=key, adjacency=adjacency)
        if name == 'gcn':
            # Both frozen GCN layers use the same graph and normalization.
            self.conv1.normalize = self.conv2.normalize = False
        return original_forward(x, cache['adjacency'])

    model.forward = types.MethodType(forward, model)
    return model
