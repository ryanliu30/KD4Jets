import torch
from torch import nn

class LorentzNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = LorentzNet(n_scalar = 2, n_hidden=72, n_layers=6, c_weight=0.005)
        self.load_state_dict(torch.load("/home/ryanliu/LorentzNet-release/logs/top/pretrained/best-val-model.pt"))
        
    def forward(self, data):
        dtype = torch.float32
        batch_size, n_nodes, _ = data['Pmu'].size()
        atom_positions = data['Pmu'].view(batch_size * n_nodes, -1).to(dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1)
        edge_mask = data['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1)
        nodes = data['nodes'].view(batch_size * n_nodes, -1).to(dtype)
        nodes = psi(nodes)
        edges = [a for a in data['edges']]
        label = data['is_signal'].to(dtype).long()
        
        return self.module(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)