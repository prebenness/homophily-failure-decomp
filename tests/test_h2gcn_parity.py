"""Math-parity test: our H2GCN encoder vs the authors' TensorFlow code.

The classes below are copied from GemsLab/H2GCN (master as of 2026-07-08),
verbatim except for import shims (their package-relative imports replaced by
locals) and the removal of code paths our setup never reaches:
- parse_network_setup and the Layer token map: h2gcn/models/__init__.py
- GCNLayer, ConcatLayer: h2gcn/models/_layers.py
- the H2GCN keras model class: h2gcn/models/H2GCN.py
- nhoodSplit and sym normalisation: h2gcn/datasets/_dataset.py (TransformSPAdj)

The test builds the authors' model for our adopted network_setup (sans MO,
dense input), copies our embedding weight into it, feeds both the same
random graph, and asserts equal outputs. This covers the hop-matrix
construction, normalisation, aggregation order, and concat layout.

Requires tensorflow (cpu build); skipped if not installed.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

tf = pytest.importorskip("tensorflow")

from hfd.models.h2gcn import H2GCNEncoder, row_normalize  # noqa: E402

keras = tf.keras


# --- verbatim: h2gcn/models/__init__.py -------------------------------------
class Layer:
    DENSE = "F"
    DROPOUT = "D"
    GCN = "G"
    RELU = "R"
    CONCAT = "C"
    VECTORIZE = "V"
    IDENTITY = "I"
    SLICE = "S"
    EXPERIMENTAL = "X"
    LAMBDA = "lambda"


def parse_network_setup(network_setup_str, output_dim,
                        _dense_units=None, _dropout_rate=None,
                        parse_preprocessing=False):
    import re
    network_setup = re.split(r"-(?![^[]*\])", network_setup_str)
    networkConf = []
    embeddingDefined = False
    for layer_str in network_setup:
        if layer_str[0] in ["F", "M"]:
            kwargs = dict()
            if len(layer_str[1:]) > 0:
                if layer_str[1:] == "O":
                    num_hidden_units = output_dim
                    kwargs["beginOutput"] = True
                else:
                    num_hidden_units = int(layer_str[1:])
            else:
                assert _dense_units is not None
                num_hidden_units = _dense_units
            if layer_str[0] == "F":
                networkConf.append(
                    (Layer.DENSE, dict(units=num_hidden_units, use_bias=True, **kwargs)))
            else:
                networkConf.append(
                    (Layer.DENSE, dict(units=num_hidden_units, use_bias=False, **kwargs)))
        elif layer_str[0] == "D":
            dropout_rate = float(layer_str[1:]) if len(layer_str[1:]) > 0 else _dropout_rate
            networkConf.append((Layer.DROPOUT, dict(dropout_rate=dropout_rate)))
        elif layer_str[0] == "G":
            hopInds = set(int(i) for i in layer_str[1:].split("_")) if len(layer_str) > 1 else None
            networkConf.append((Layer.GCN, dict(hops=hopInds)))
        elif layer_str[0] == "C":
            tags = [i for i in layer_str[1:].split("_")]
            networkConf.append((Layer.CONCAT, dict(tags=tags, addInputs=True)))
        elif layer_str[0] == "R":
            networkConf.append((Layer.RELU, dict()))
        elif layer_str[0] == "V":
            networkConf.append((Layer.VECTORIZE, dict()))
        elif layer_str[0] == "T":
            networkConf[-1][-1]["tag"] = layer_str[1:]
        else:
            raise ValueError(f"Unknown layer config {layer_str}")
    return networkConf


# --- verbatim: h2gcn/models/_layers.py ---------------------------------------
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, hops=None):
        super().__init__()
        self.hops = hops

    def call(self, adjhops, inputs):
        return tf.stack(
            [tf.sparse.sparse_dense_matmul(x, inputs) for ind, x in enumerate(adjhops)
             if (self.hops is None or ind in self.hops)], axis=-2)


class ConcatLayer(tf.keras.layers.Layer):
    def __init__(self, tags, axis=-1, addInputs=True):
        super().__init__()
        self.tags = tags
        self.axis = axis
        self.addInputs = addInputs

    def call(self, *args, **kwargs):
        selected = [value for name, value in kwargs.items() if name in self.tags]
        if self.addInputs:
            return tf.concat(list(args) + selected, self.axis)
        else:
            return tf.concat(selected, self.axis)


# --- verbatim (trimmed to reachable paths): h2gcn/models/H2GCN.py ------------
class H2GCN(tf.keras.Model):
    def __init__(self, layer_setups, sparse_input=True, l2_regularize_weight=0):
        super().__init__()
        self.layer_objs = []
        self.dropout_inds = []
        self.graph_hops_inds = []
        self.concat_inds = []
        self.tagsDict = dict()

        for ind, (layerType, layerConf) in enumerate(layer_setups):
            layerTag = layerConf.pop("tag", None)
            if layerType == Layer.DENSE:
                self.layer_objs.append(keras.layers.Dense(
                    layerConf["units"],
                    use_bias=layerConf["use_bias"],
                    kernel_regularizer=keras.regularizers.l2(l2_regularize_weight)
                ))
            elif layerType == Layer.DROPOUT:
                self.dropout_inds.append(len(self.layer_objs))
                self.layer_objs.append(keras.layers.Dropout(layerConf["dropout_rate"]))
            elif layerType == Layer.GCN:
                self.graph_hops_inds.append(len(self.layer_objs))
                self.layer_objs.append(GCNLayer(**layerConf))
            elif layerType == Layer.RELU:
                self.layer_objs.append(tf.keras.layers.ReLU())
            elif layerType == Layer.VECTORIZE:
                self.layer_objs.append(tf.keras.layers.Flatten())
            elif layerType == Layer.CONCAT:
                self.concat_inds.append(len(self.layer_objs))
                self.layer_objs.append(ConcatLayer(
                    tags=layerConf["tags"], addInputs=layerConf["addInputs"]))
            else:
                raise ValueError(f"Unsupported layer type {layerType}")
            if layerTag:
                self.tagsDict[len(self.layer_objs) - 1] = layerTag

    def call(self, adj, inputs, adjhops, training=False):
        taggedOutputs = dict()
        for ind, layer in enumerate(self.layer_objs):
            if ind in self.concat_inds:
                inputs = layer(inputs, **taggedOutputs)
            elif ind in self.graph_hops_inds:
                inputs = layer(adjhops, inputs)
            else:
                inputs = layer(inputs)
            if ind in self.tagsDict:
                taggedOutputs[self.tagsDict[ind]] = inputs
        return inputs


# --- verbatim: h2gcn/datasets/_dataset.py (TransformSPAdj) --------------------
def nhoodSplit(adj, nhood):
    mt = sp.eye(adj.shape[1])
    mtList = [mt]
    i = 0
    edge_sum = 0
    while i < nhood:
        prev_mt = mt
        mt = mt @ (adj + sp.eye(adj.shape[0]))
        mt = (mt > 0).astype(mt.dtype)
        new_edge_sum = mt.sum()
        if edge_sum == new_edge_sum:
            break
        else:
            edge_sum = new_edge_sum
        i += 1
        mtList.append(mt - prev_mt)
    return mtList


def sym_normalize(adj):
    degInvSqrt = np.power(adj.sum(axis=1).A1, -0.5)
    degInvSqrt[np.isinf(degInvSqrt)] = 0.
    DInvSqrt = sp.diags(degInvSqrt)
    return DInvSqrt @ adj @ DInvSqrt


def to_tf_sparse(spmat):
    X = spmat.tocoo()
    indices = np.array([X.row, X.col]).T
    return tf.sparse.reorder(
        tf.SparseTensor(indices, X.data.astype(np.float32), X.shape))


# --- the parity test ----------------------------------------------------------
def test_h2gcn_encoder_matches_reference_tf():
    rng = np.random.RandomState(3)
    n, f, hidden = 14, 6, 8
    # Random simple undirected graph, no self-loops.
    dense = (rng.rand(n, n) < 0.25).astype(np.float64)
    dense = np.triu(dense, 1)
    dense = dense + dense.T
    adj = sp.csr_matrix(dense)
    x_np = rng.rand(n, f).astype(np.float32)
    edge_index = np.array(adj.nonzero())

    # Authors' pipeline: hop split on the raw adjacency, sym-normalised.
    splits = nhoodSplit(adj, 2)
    adjhops = [to_tf_sparse(sp.csr_matrix(sym_normalize(splits[1]))),
               to_tf_sparse(sp.csr_matrix(sym_normalize(splits[2])))]

    setups = parse_network_setup(
        f"M{hidden}-R-T1-G-V-T2-G-V-C1-C2-D0.5", output_dim=None)
    reference = H2GCN(setups, sparse_input=False)

    # Row-normalised features go in (their preprocessing step).
    x_norm = row_normalize(torch.from_numpy(x_np)).numpy()
    _ = reference(None, tf.constant(x_norm), adjhops, training=False)  # build

    ours = H2GCNEncoder(in_dim=f, hidden_dim=hidden)
    ours.eval()
    # Copy our embedding weight into their dense kernel (keras: [in, out]).
    kernel = ours.embed.weight.detach().numpy().T
    reference.layer_objs[0].set_weights([kernel])

    out_ref = reference(None, tf.constant(x_norm), adjhops, training=False).numpy()
    with torch.no_grad():
        out_ours = ours(torch.from_numpy(x_np), torch.from_numpy(edge_index)).numpy()

    assert out_ours.shape == out_ref.shape == (n, 7 * hidden)
    assert np.allclose(out_ours, out_ref, atol=1e-5), (
        f"max abs diff {np.abs(out_ours - out_ref).max()}"
    )
