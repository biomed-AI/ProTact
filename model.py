import torch
import dgl

import torch.nn.functional as F

from torch import nn
from e3nn import o3
from e3nn.nn import BatchNorm
from torch_scatter import scatter


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.reshape(-1, 1) - self.offset.reshape(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class I3NN(nn.Module):
    def __init__(
            self,
            edge_input_dim=1,
            sh_lmax=2,
            ns=128,
            nv=32,
            num_conv_layers=2,
            max_radius=30,
            distance_embed_dim=32,
            use_second_order_repr=False,
            batch_norm=True,
            dropout=0.0,
    ):
        super(I3NN, self).__init__()
        self.ns = ns
        self.nv = nv
        self.distance_expansion = GaussianSmearing(0.0, max_radius, distance_embed_dim)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.edge_embedding = nn.Sequential(
                nn.Linear(edge_input_dim + distance_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
        )

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {ns}x0o + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {ns}x0o + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                # f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {ns}x0o + {nv}x1o',
                f'{ns}x0e + {ns}x0o + {nv}x1o + {nv}x1e',
                # f'{ns}x0e + {ns}x0o + {nv}x1o + {nv}x1e'
            ]
        conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            layer = TensorProductConvLayer(**parameters)
            conv_layers.append(layer)
        conv_layers.append(TensorProductConvLayer(
                in_irreps=irrep_seq[num_conv_layers],
                sh_irreps=self.sh_irreps,
                out_irreps=f'{ns}x0e + {ns}x0o',
                n_edge_features=3*ns,
                hidden_features=3*ns,
                residual=False,
                batch_norm=batch_norm,
                dropout=dropout,
        ))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.node_output_layer = nn.Linear(2 * self.ns, ns)
        self.edge_output_layer = nn.Linear(3 * self.ns, ns)

    def forward(self, graph: dgl.DGLGraph):
        node_attr, edge_index, edge_attr, edge_sh = self.build_conv_graph(graph)
        src, dst = edge_index
        edge_attr = self.edge_embedding(edge_attr)
        for layer in self.conv_layers:
            edge_attr_augment = torch.cat([
                edge_attr,
                node_attr[src, :self.ns],
                node_attr[dst, :self.ns],
            ], dim=-1)
            node_update = layer(node_attr, edge_index, edge_attr_augment, edge_sh)
            node_attr = F.pad(node_attr, (0, node_update.shape[-1] - node_attr.shape[-1]))
            node_attr += node_update
        graph.ndata["f"] = self.node_output_layer(node_attr)
        graph.edata["f"] = self.edge_output_layer(edge_attr_augment)
        return graph

    def build_conv_graph(self, graph: dgl.DGLGraph):
        # builds the receptor initial node and edge embeddings
        node_attr = graph.ndata['f']

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = graph.edges()
        src, dst = edge_index
        edge_vec = graph.ndata['x'][dst.long()] - graph.ndata['x'][src.long()]

        edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = torch.cat([graph.edata["f"], edge_length_emb], dim=-1)
        edge_vec_norm = F.normalize(edge_vec, p=2, dim=-1)
        alpha = torch.acos((graph.ndata['nuv'][dst.long()][:, 0, :] * edge_vec_norm).sum(-1).clamp(-1, 1))
        beta = torch.acos((graph.ndata['nuv'][src.long()][:, 0, :] * edge_vec_norm).sum(-1).clamp(-1, 1))
        # edge_invariant_vec = o3.angles_to_xyz(alpha, beta)
        edge_sh = o3.spherical_harmonics_alpha_beta(self.sh_irreps, alpha, beta, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh


class TrigonometryUpdate(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)

        self.gate_linear1 = nn.Linear(embedding_channels, c)
        self.gate_linear2 = nn.Linear(embedding_channels, c)

        self.linear1 = nn.Linear(embedding_channels, c)
        self.linear2 = nn.Linear(embedding_channels, c)

        # self.gate_linear_z = nn.Linear(embedding_channels, c)
        # self.gate_linear = nn.Linear(embedding_channels, c)
        #
        # self.linear_z = nn.Linear(embedding_channels, c)
        # self.linear = nn.Linear(embedding_channels, c)

        self.ending_gate_linear = nn.Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = nn.Linear(c, embedding_channels)

    def forward(self, z, graph1: dgl.DGLGraph, graph2: dgl.DGLGraph):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        protein1_pair = self.layernorm(graph1.edata["f"])
        protein2_pair = self.layernorm(graph2.edata["f"])

        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z)
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z)
        protein1_pair = self.gate_linear1(protein1_pair).sigmoid() * self.linear1(protein1_pair)
        protein2_pair = self.gate_linear2(protein2_pair).sigmoid() * self.linear2(protein2_pair)

        # ab = self.gate_linear_z(z).sigmoid() * self.linear_z(z)
        # # ab = self.gate_linear(z).sigmoid() * self.linear(z)
        # protein1_pair = self.gate_linear(protein1_pair).sigmoid() * self.linear(protein1_pair)
        # protein2_pair = self.gate_linear(protein2_pair).sigmoid() * self.linear(protein2_pair)

        g = self.ending_gate_linear(z).sigmoid()
        protein1_pair_coo = torch.sparse.FloatTensor(torch.stack(graph1.edges()).long(), protein1_pair).to_dense()
        protein2_pair_coo = torch.sparse.FloatTensor(torch.stack(graph2.edges()).long(), protein2_pair).to_dense()
        block1 = torch.einsum("...ikc,...kjc->...ijc", protein1_pair_coo, ab1)
        block2 = torch.einsum("...ikc,...jkc->...ijc", ab2, protein2_pair_coo)
        # block1 = torch.einsum("...ikc,...kjc->...ijc", protein1_pair_coo, ab)
        # block2 = torch.einsum("...ikc,...jkc->...ijc", ab, protein2_pair_coo)
        # print(g.shape, block1.shape, block2.shape)
        z = g * self.linear_after_sum(self.layernorm_c(block1 + block2))
        return z


class TriangleSelfAttentionRowWise(torch.nn.Module):
    # use the protein-compound matrix only.
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self.dp = nn.Dropout(drop_rate)
        # self.ln = nn.LayerNorm(hidden_size)

        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        # self.layernorm_c = torch.nn.LayerNorm(c)

        self.linear_q = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        # self.b = Linear(embedding_channels, h, bias=False)
        self.g = nn.Linear(embedding_channels, self.all_head_size)
        self.final_linear = nn.Linear(self.all_head_size, embedding_channels)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        return x

    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask of shape b, i, j
        z = self.layernorm(z)
        # new_z = torch.zeros(z.shape, device=z.device)
        z_i = z
        # q, k, v of shape b, j, h, c
        q = self.reshape_last_dim(self.linear_q(z_i))  #  * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z_i))
        v = self.reshape_last_dim(self.linear_v(z_i))
        logits = torch.einsum("...iqhc,...ikhc->...ihqk", q, k)
        weights = nn.Softmax(dim=-1)(logits)
        # weights of shape b, h, j, j
        # attention_probs = self.dp(attention_probs)
        weighted_avg = torch.einsum("...ihqk,...ikhc->...iqhc", weights, v)
        g = self.reshape_last_dim(self.g(z_i)).sigmoid()
        output = g * weighted_avg
        new_output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.reshape(*new_output_shape)
        # output of shape b, j, embedding.
        # z[:, i] = output
        z = output
        # print(g.shape, block1.shape, block2.shape)
        z = self.final_linear(z)
        return z


class Transition(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, input_embedding_channels=256, embedding_channels=256, n=4):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(input_embedding_channels)
        self.linear1 = nn.Linear(input_embedding_channels, n * embedding_channels)
        self.linear2 = nn.Linear(n * embedding_channels, embedding_channels)

    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z


class TriangleModule(nn.Module):
    def __init__(self,
                 n_trigonometry_module_stack: int = 2,
                 embedding_channels: int = 256,
                 c: int = 16,
                 num_classes: int = 2,
                 output_emb=False):
        super(TriangleModule, self).__init__()
        self.output_emb = output_emb
        self.dropout = nn.Dropout2d(p=0.25)
        self.n_trigonometry_module_stack = n_trigonometry_module_stack
        self.trigonometry_update_list = nn.ModuleList([
            TrigonometryUpdate(embedding_channels=embedding_channels, c=c)
            for _ in range(n_trigonometry_module_stack)
        ])
        self.triangle_self_attention_list = nn.ModuleList([
            TriangleSelfAttentionRowWise(embedding_channels=embedding_channels)
            for _ in range(n_trigonometry_module_stack)
        ])
        self.tranistion = Transition(input_embedding_channels=embedding_channels,
                                     embedding_channels=embedding_channels)

        self.output = nn.Linear(embedding_channels, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        final_layer_bias = self.output.bias.clone()
        final_layer_bias[-1] = -7.0  # -7 chosen as the second term's bias s.t. positives are predicted with prob=0.001
        self.output.bias = nn.Parameter(final_layer_bias, requires_grad=True)

    def forward(self, graph1: dgl.DGLGraph, graph2: dgl.DGLGraph):
        z = torch.einsum("...ik,...jk->...ijk", graph1.ndata["f"], graph2.ndata["f"])
        for i_module in range(self.n_trigonometry_module_stack):
            z = z + self.dropout(
                    self.trigonometry_update_list[i_module](
                            z, graph1, graph2
                    )
            )
            z = z + self.dropout(self.triangle_self_attention_list[i_module](z))
            z = self.tranistion(z)
        if self.output_emb:
            return self.output(z), z
        else:
            return self.output(z)


class ProTact(nn.Module):
    def __init__(
            self,
            num_node_input_feats=113,
            num_edge_input_feats=28,
            num_classes=2,
            num_gnn_layers=2,
            num_gnn_hidden_channels=128,
            num_interact_layers=7,
            output_emb=False):
        super().__init__()
        self.output_emb = output_emb
        # Build the network
        self.num_node_input_feats = num_node_input_feats
        self.num_edge_input_feats = num_edge_input_feats
        self.num_classes = num_classes
        # GNN module's keyword arguments provided via the command line
        self.num_gnn_layers = num_gnn_layers
        self.num_gnn_hidden_channels = num_gnn_hidden_channels

        # Interaction module's keyword arguments provided via the command line
        self.num_interact_layers = num_interact_layers
        self.num_interact_hidden_channels = num_gnn_hidden_channels

        # Model hyperparameter keyword arguments provided via the command line
        self.origin_features = 113
        self.other_embedding = self.num_node_input_feats > self.origin_features
        if self.other_embedding:
            self.node_in_embedding = nn.Linear(self.origin_features, int(self.num_gnn_hidden_channels / 2), bias=False)
            self.node_other_in_embedding = nn.Linear(self.num_node_input_feats - self.origin_features, int(self.num_gnn_hidden_channels/2), bias=False)
        else:
            self.node_in_embedding = nn.Linear(self.num_node_input_feats, self.num_gnn_hidden_channels, bias=False)

        if self.num_gnn_layers > 0:
            gnn_modules = [I3NN(
                    edge_input_dim=self.num_edge_input_feats,
                    ns=self.num_gnn_hidden_channels,
                    num_conv_layers=self.num_gnn_layers,
            )]
        else:
            gnn_modules = []

        self.gnn_module = nn.ModuleList(gnn_modules)
        self.interact_module = TriangleModule(
                embedding_channels=self.num_interact_hidden_channels,
                c=128,
                n_trigonometry_module_stack=self.num_interact_layers,
                num_classes=self.num_classes,
                output_emb=self.output_emb
        )

    def gnn_forward(self, graph: dgl.DGLGraph):
        """Make a forward pass through a single GNN module."""
        # Embed input features a priori
        if self.other_embedding:
            graph.ndata['f'] = torch.cat([
                self.node_in_embedding(graph.ndata['f'][:, :self.origin_features]).squeeze(),
                self.node_other_in_embedding(graph.ndata['f'][:, self.origin_features:]).squeeze(),
            ], dim=-1)
        else:
            graph.ndata['f'] = self.node_in_embedding(graph.ndata['f']).squeeze()
        for layer in self.gnn_module:
            graph = layer(graph)  # Geometric Transformers can handle their own depth
        # Unbatch and collect each individual data's predicted node features
        graphs = dgl.unbatch(graph)
        node_feats = [graph.ndata['f'] for graph in graphs]
        return node_feats

    def forward(self, graph1, graph2):
        graph1_node_feats = self.gnn_forward(graph1)
        graph2_node_feats = self.gnn_forward(graph2)

        if self.output_emb:
            interact_tensor, z = self.interact_module(graph1, graph2)
            return [interact_tensor, graph1_node_feats[0], graph2_node_feats[0], z]
        else:
            interact_tensor = self.interact_module(graph1, graph2)
            return [interact_tensor]
