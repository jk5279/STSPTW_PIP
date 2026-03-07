"""
Transformer encoder building blocks shared by POMO-based models.
SINGLE_Encoder, EncoderLayer, utility functions / modules.

Sourced from jk5279/STSPTW_PIP — purely architectural (no env-specific logic).
OTAModel uses: SINGLE_Encoder, EncoderLayer, reshape_by_heads,
               multi_head_attention, Add_And_Normalization_Module, FeedForward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TSPTW_PROBLEMS = frozenset({"TSPTW", "TSPTW_SPIP"})

__all__ = [
    'SINGLE_Encoder', 'EncoderLayer',
    'reshape_by_heads', 'multi_head_attention',
    'Add_And_Normalization_Module', 'FeedForward',
]


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class SINGLE_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        if not self.problem.startswith("TSP"):
            self.embedding_depot = nn.Linear(2, embedding_dim)
        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL",
                            "OVRPB", "OVRPL", "OVRPBL"]:
            self.embedding_node = nn.Linear(3, embedding_dim)
        elif self.problem in TSPTW_PROBLEMS or self.problem == "TSPDL":
            self.embedding_node = nn.Linear(4, embedding_dim)
        elif self.problem in ["VRPTW", "OVRPTW", "VRPBTW", "VRPLTW",
                              "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
            self.embedding_node = nn.Linear(5, embedding_dim)
        else:
            raise NotImplementedError(f"Unsupported problem: {self.problem}")

        self.layers = nn.ModuleList(
            [EncoderLayer(**model_params) for _ in range(encoder_layer_num)]
        )

    def forward(self, depot_xy, node_xy_demand_tw):
        # depot_xy.shape: (batch, 1, 2)  — None for TSP variants
        # node_xy_demand_tw.shape: (batch, problem, 3/4/5)
        if depot_xy is not None:
            embedded_depot = self.embedding_depot(depot_xy)   # (batch, 1, emb)
        embedded_node = self.embedding_node(node_xy_demand_tw)  # (batch, N, emb)

        if depot_xy is not None:
            out = torch.cat((embedded_depot, embedded_node), dim=1)
        else:
            out = embedded_node

        for layer in self.layers:
            out = layer(out)

        return out  # (batch, problem(+1), embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem, embedding_dim)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q/k/v shape: (batch, HEAD_NUM, problem, KEY_DIM)

        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)
            multi_head_out = self.multi_head_combine(out_concat)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)
        else:
            # norm_first convention
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(
                multi_head_attention(
                    reshape_by_heads(self.Wq(out1), head_num=head_num),
                    reshape_by_heads(self.Wk(out1), head_num=head_num),
                    reshape_by_heads(self.Wv(out1), head_num=head_num),
                )
            )
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2

        return out3


# ─────────────────────────────────────────────────────────────────────────────
# NN SUB-CLASSES / FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def reshape_by_heads(qkv, head_num):
    # qkv.shape: (batch, n, head_num*key_dim)
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)   # (batch, n, H, K)
    return q_reshaped.transpose(1, 2)                     # (batch, H, n, K)


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q/k/v shape: (batch, head_num, n, key_dim)
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))   # (batch, H, n, problem)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(
            batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(
            batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)       # (batch, H, n, problem)
    out = torch.matmul(weights, v)                  # (batch, H, n, key_dim)
    out_transposed = out.transpose(1, 2)            # (batch, n, H, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat                               # (batch, n, H*key_dim)


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = (
            'norm_loc' in model_params
            and model_params['norm_loc'] == "norm_last"
        )
        norm_type = model_params.get("norm", "batch")
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True,
                                       track_running_stats=True)
        elif norm_type == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True,
                                       track_running_stats=False)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True,
                                          track_running_stats=False)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif norm_type == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]),
                                           requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = (input1 + input2) if self.add else input2
            back_trans = self.norm(added.transpose(1, 2)).transpose(1, 2)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = (input1 + input2) if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = (input1 + input2) if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = (input1 + self.norm * input2) if self.add \
                else (self.norm * input2)
        else:
            back_trans = (input1 + input2) if self.add else input2
        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
