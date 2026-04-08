# Blueprint: DGL → PyTorch Geometric (PyG) Migration

**Goal:** Replace all DGL dependencies with PyG equivalents while preserving model behavior and improving cross-platform compatibility (Windows/macOS).

---

## Why Migrate

- DGL dropped Windows/macOS support in recent versions
- PyG (`torch_geometric`) is actively maintained, pip-installable on all platforms
- PyG has native implementations for every GNN layer used (GGNN, PNA, GINE)
- PyG's `MessagePassing` base class is more Pythonic than DGL's `update_all` pattern
- PyG integrates better with modern PyTorch (torch.compile, torch_sparse, etc.)

---

## Scope

19 files import DGL. All 19 need changes. The migration is layered:

```
Layer 1: nn_utils/ (core graph primitives)          ← migrate first
Layer 2: */data.py files (graph construction)       ← depends on Layer 1
Layer 3: */model.py files (GNN layers & forward)    ← depends on Layers 1-2
Layer 4: environment.yml / requirements.txt         ← swap packages last
```

---

## API Translation Reference

### Graph Object

| DGL | PyG | Notes |
|-----|-----|-------|
| `dgl.DGLGraph` | `torch_geometric.data.Data` | Core graph object |
| `g.ndata["h"]` | `data.x` | Node features |
| `g.edata["e"]` | `data.edge_attr` | Edge features |
| `g.edata["e_ind"]` | `data.edge_type` | Edge type indices |
| `g.num_nodes()` | `data.num_nodes` | Node count |
| `g.num_edges()` | `data.num_edges` | Edge count |
| `dgl.graph((src, dst))` | `Data(edge_index=torch.stack([src, dst]))` | Edge index is `(2, E)` in PyG |

> **Critical shape difference:** DGL edge lists are `(E, 2)`. PyG `edge_index` is `(2, E)`.

### Batching

| DGL | PyG | Notes |
|-----|-----|-------|
| `dgl.batch(graphs)` | `Batch.from_data_list(graphs)` | Merges graphs |
| `dgl.unbatch(batch)` | `batch.to_data_list()` | Splits back |
| `batch.batch_size` | `batch.num_graphs` | Number of graphs |
| `batch.batch_num_nodes()` | `torch.bincount(batch.batch)` | Per-graph node counts |
| `batch.batch_num_edges()` | `torch.bincount(batch_edge)` | Per-graph edge counts (derived) |
| `embed.repeat_interleave(batch.batch_num_nodes(), 0)` | `embed[batch.batch]` | Broadcast graph-level to node-level |

The `batch.batch` tensor is shape `(N_total,)` mapping each node to its graph index — used everywhere DGL used `batch_num_nodes()`.

### Pooling

| DGL | PyG |
|-----|-----|
| `dgl_nn.AvgPooling()(g, h)` | `global_mean_pool(h, batch.batch)` |
| `dgl_nn.SumPooling()(g, h)` | `global_add_pool(h, batch.batch)` |
| `dgl_nn.GlobalAttentionPooling(gate_nn)(g, h)` | `GlobalAttention(gate_nn)(h, batch.batch)` |

### GNN Layers

| DGL Custom Layer | PyG Equivalent | Notes |
|-----------------|----------------|-------|
| `GatedGraphConv` | `torch_geometric.nn.GatedGraphConv` | Direct drop-in; same semantics |
| `PNAConv` | `torch_geometric.nn.PNAConv` | Native PyG; needs `deg` histogram |
| `GINEConv` | `torch_geometric.nn.GINEConv` | Native PyG; pass `nn` and `edge_dim` |
| `GCNLayer` | `torch_geometric.nn.GCNConv` | Standard GCN |
| `MultiEdgeGCNLayer` | Manual loop over `torch_geometric.nn.GCNConv` per type | No direct hetero equivalent needed |
| `dgl_nn.GraphConv` | `torch_geometric.nn.GCNConv` | `add_self_loops=False` to match |

### Positional Encoding

| DGL | PyG |
|-----|-----|
| Custom `random_walk_pe(g, k)` | `torch_geometric.transforms.AddRandomWalkPE(walk_length=k)` |

PyG's transform integrates directly into `Data` objects and can be applied in the dataset.

### Message Passing (Custom Layers)

DGL pattern:
```python
def message(edges):
    return {"msg": F.relu(edges.src["h"] + edges.data["e"])}
def reduce(nodes):
    return {"h_neigh": nodes.mailbox["msg"].sum(dim=1)}
graph.update_all(message, reduce)
```

PyG equivalent — subclass `MessagePassing`:
```python
from torch_geometric.nn import MessagePassing

class CustomLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")  # or "mean", "max"

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)  # x_j = source node features
```

---

## File-by-File Migration Plan

### Layer 1: `src/ms_pred/nn_utils/`

#### `mol_graph.py` → rewrite `get_dgl_graph()`

Current output:
```python
g = dgl.graph((src, dst), num_nodes=N)
g.ndata["h"] = atom_feats   # (N, 86)
g.edata["e"] = bond_feats   # (E, 5)
```

New output:
```python
from torch_geometric.data import Data

edge_index = torch.tensor([src, dst], dtype=torch.long)   # (2, E)
data = Data(
    x=torch.tensor(atom_feats, dtype=torch.float),         # (N, 86)
    edge_index=edge_index,
    edge_attr=torch.tensor(bond_feats, dtype=torch.float), # (E, 5)
)
```

For bidirectional graphs — double edges (same as before, just stack row-wise on edge_index):
```python
rev_edge_index = edge_index.flip(0)
edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
```

#### `dgl_modules.py` → replace all custom layers

Replace custom `GatedGraphConv`, `PNAConv`, `GINEConv`, `GCNLayer` with PyG natives.
Only `MultiEdgeGCNLayer` (heterograph) needs a manual rewrite as a loop of `GCNConv` layers.

For `PNAConv`, the `deg` histogram must be precomputed from the training dataset:
```python
from torch_geometric.utils import degree
deg = torch.zeros(max_degree + 1, dtype=torch.long)
for data in train_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())
```
Store `deg` in dataset and pass to model constructor.

#### `nn_utils.py` → update `MoleculeGNN`, `GGNN`, `PNA`, `GINE`, utilities

- `MoleculeGNN.forward(graph, feat)` → `MoleculeGNN.forward(data)` using `data.x`, `data.edge_index`, `data.edge_attr`, `data.batch`
- `split_dgl_batch()` → rewrite using `batch.to_data_list()` + `Batch.from_data_list()`
- `random_walk_pe()` → replace with `AddRandomWalkPE` transform or keep function signature using `torch_geometric.utils.get_laplacian` internals
- `pad_packed_tensor` / `pack_padded_tensor` → replace with `torch.bincount(batch.batch)` + scatter ops

---

### Layer 2: `*/dag_data.py`, `*/scarf_data.py`, `*/gnn_data.py`, etc.

Each data file that calls `dgl.batch()` in a `collate_fn` needs:

```python
# Before
from torch.utils.data import DataLoader
collate = lambda batch: {"graphs": dgl.batch([b["graph"] for b in batch])}

# After
from torch_geometric.data import Batch
collate = lambda batch: {"graphs": Batch.from_data_list([b["graph"] for b in batch])}
```

For DAG models, `frag_atoms` is currently derived from `graph.batch_num_nodes()`.
Replace with:
```python
frag_atoms = torch.bincount(frag_batch.batch)
```

Root-to-fragment index mapping (`root_inds`, scatter operations) needs to use
`frag_batch.batch` as the scatter index instead of `repeat_interleave(batch_num_nodes)`.

---

### Layer 3: `*/gen_model.py`, `*/inten_model.py`, `*/gnn_model.py`, etc.

**Adduct/metadata broadcast pattern change:**

```python
# DGL pattern
embed_adducts_expand = embed_adducts.repeat_interleave(graphs.batch_num_nodes(), 0)

# PyG pattern
embed_adducts_expand = embed_adducts[graphs.batch]
```

**Graph-level readout:**
```python
# DGL
out = self.pool(graphs, node_h)

# PyG
from torch_geometric.nn import global_mean_pool, GlobalAttention
out = global_mean_pool(node_h, graphs.batch)
```

**DAG fragment-level model (gen_model.py, inten_model.py):**

The most complex migration. Fragment graphs are batched and processed together, then results are scattered back to root molecules.

```python
# DGL
frag_batch = dgl.batch(frag_graphs)
frag_h = self.frag_gnn(frag_batch, frag_batch.ndata["h"])
# Pool per fragment
frag_pooled = self.pool(frag_batch, frag_h)  # (K, hidden)

# PyG
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

frag_batch = Batch.from_data_list(frag_graphs)
frag_h = self.frag_gnn(frag_batch)           # node-level
frag_pooled = global_mean_pool(frag_h, frag_batch.batch)  # (K, hidden)
```

---

### Layer 4: Environment & Dependencies

**`environment.yml` changes:**
```yaml
# Remove
- dgl=0.8.2
- dgl-cuda11.1

# Add
- pyg  # or: pip: [torch_geometric]
```

**`requirements.txt` changes:**
```
# Remove
dgl

# Add
torch_geometric
torch_scatter
torch_sparse
torch_cluster
```

Install PyG properly (platform-independent):
```bash
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-<VERSION>+<CUDA>.html
```

---

## Architecture Improvement Opportunities

While migrating, these improvements are worth considering:

### 1. Unify PNA degree histogram precomputation
Currently not explicitly done — just passed as a config int. Switch to the proper PyG pattern of computing `deg` from training data for correct PNA scaling.

### 2. Use PyG's built-in transforms for preprocessing
- `AddRandomWalkPE` — replaces manual `random_walk_pe()`
- `AddLaplacianEigenvectorPE` — alternative PE for better structural awareness
- `ToUndirected()` — replaces manual edge doubling

Apply transforms at dataset construction time, not at batch collation.

### 3. Use `torch_geometric.loader.DataLoader`
Replaces standard PyTorch DataLoader with custom collate. Handles batching automatically:
```python
from torch_geometric.loader import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Each batch is already a Batch object with .batch, .num_graphs, etc.
```

### 4. Use `HeteroData` for fragment-root relationships (optional)
The DAG fragment-to-root mapping is currently handled via external index tensors. PyG's `HeteroData` could encode root/fragment as different node types with explicit edges — cleaner than manual scatter index tracking.

### 5. `torch.compile` compatibility
After migration, all PyG ops are compatible with `torch.compile`, enabling potential inference speedup. No DGL ops were compatible.

---

## Migration Checkpoints

After each layer, verify with:

1. **Layer 1 (nn_utils):** Unit test `get_dgl_graph` output — check `data.x.shape`, `data.edge_index.shape`, `data.edge_attr.shape`
2. **Layer 2 (data files):** Run a single batch through the DataLoader — check batch shapes match pre-migration
3. **Layer 3 (models):** Run a single training step — check loss is finite and not NaN
4. **Layer 4 (environment):** Fresh env install — confirm `import torch_geometric` works and no DGL imports remain (`grep -r "import dgl" src/`)

---

## Files to Modify (Ordered)

```
Priority 1 (blockers for everything else):
  src/ms_pred/nn_utils/mol_graph.py
  src/ms_pred/nn_utils/dgl_modules.py
  src/ms_pred/nn_utils/nn_utils.py

Priority 2 (data pipeline):
  src/ms_pred/gnn_pred/gnn_data.py
  src/ms_pred/dag_pred/dag_data.py
  src/ms_pred/marason/dag_data.py
  src/ms_pred/scarf_pred/scarf_data.py
  src/ms_pred/graff_ms/graff_ms_data.py
  src/ms_pred/autoregr_gen/autoregr_data.py
  src/ms_pred/molnetms/molnetms_data.py
  src/ms_pred/massformer_pred/massformer_data.py

Priority 3 (models):
  src/ms_pred/gnn_pred/gnn_model.py
  src/ms_pred/dag_pred/gen_model.py
  src/ms_pred/dag_pred/inten_model.py
  src/ms_pred/marason/gen_model.py
  src/ms_pred/marason/inten_model.py
  src/ms_pred/scarf_pred/scarf_model.py
  src/ms_pred/graff_ms/graff_ms_model.py
  src/ms_pred/autoregr_gen/autoregr_model.py

Priority 4 (environment):
  environment.yml
  requirements.txt
  setup.py (if referencing dgl)
```

---

## Risks & Notes

- **`local_scope()`**: DGL's context manager for temporary graph features has no PyG equivalent — use Python variables instead (PyG layers don't mutate graph state).
- **`split_dgl_batch`**: Dynamic batch splitting logic must be rewritten using `Batch.to_data_list()`. The edge-count threshold logic remains the same.
- **MAGMA fragment graphs**: `dag_data.py`'s `dgl_featurize()` method is the single source of fragment graph construction — migrating this one method cascades through both `dag_pred` and `marason`.
- **`massformer_pred`**: MassFormer uses its own Cython modules. Only the DGL graph construction in `massformer_data.py` needs migration, not the model itself.
- **marason + pygmtools**: MARASON uses PyGMTools for graph matching alongside DGL. PyGMTools has a PyG backend — switch `backend="pytorch_geometric"` in graph matching calls.
