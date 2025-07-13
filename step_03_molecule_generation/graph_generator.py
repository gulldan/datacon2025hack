"""Graph-based generator (initial version).

This *very lightweight* implementation trains a graph encoder (GCN-based VAE)
that embeds molecules into latent space but **does not yet include a full
learned decoder**.  For sampling we:
1. Draw latent vectors from the standard normal.
2. Find nearest neighbour (cosine) among training molecule latents.
3. Return the corresponding SMILES.

Consequently the output set is a re-weighted subset of the training data – a
reasonable placeholder until a full Graph Residual Flow model is implemented.

Dependencies: ``torch`` and ``torch_geometric``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from rdkit import Chem  # type: ignore
from torch import nn
from torch_geometric.data import Data, DataLoader  # type: ignore
from torch_geometric.nn import GCNConv, global_mean_pool  # type: ignore

import config
from utils.logger import LOGGER

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Graph dataset utilities
# ---------------------------------------------------------------------------

ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # common heavy atoms
ATOM_TO_IDX = {z: i for i, z in enumerate(ATOM_TYPES)}


def mol_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    # Node features: one-hot atom type
    x = []
    for atom in mol.GetAtoms():  # type: ignore[attr-defined]
        z = atom.GetAtomicNum()  # type: ignore[attr-defined]
        onehot = [0] * len(ATOM_TYPES)
        if z in ATOM_TO_IDX:
            onehot[ATOM_TO_IDX[z]] = 1
        x.append(onehot)
    x = torch.tensor(x, dtype=torch.float)

    # Edges: undirected bonds
    edge_index = []
    for bond in mol.GetBonds():  # type: ignore[attr-defined]
        i = bond.GetBeginAtomIdx()  # type: ignore[attr-defined]
        j = bond.GetEndAtomIdx()  # type: ignore[attr-defined]
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, smiles=smiles)


# ---------------------------------------------------------------------------
# Simple Graph VAE encoder (no decoder yet)
# ---------------------------------------------------------------------------

LATENT_DIM = 64


class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(len(ATOM_TYPES), 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)
        self.lin_mu = nn.Linear(256, LATENT_DIM)
        self.lin_logvar = nn.Linear(256, LATENT_DIM)

    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x, data.edge_index, data.batch  # type: ignore[attr-defined]
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin_mu(x), self.lin_logvar(x)


# ---------------------------------------------------------------------------
# Training helper – optimise KL only (reconstruction skipped)
# ---------------------------------------------------------------------------


def train_encoder(dataset: list[Data]):
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    enc = GraphEncoder().to(DEV)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    for epoch in range(1, 11):  # 10 epochs is enough for placeholder
        enc.train()
        tot = 0.0
        for batch in loader:
            batch = batch.to(DEV)
            opt.zero_grad()
            mu, logvar = enc(batch)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kld.backward()
            opt.step()
            tot += kld.item() * batch.num_graphs  # type: ignore[attr-defined]
        LOGGER.info(f"GraphEncoder epoch {epoch} – KL {tot / len(dataset):.4f}")
    return enc


# ---------------------------------------------------------------------------
# Public API – compatible with SELFIES generator
# ---------------------------------------------------------------------------


def train_and_sample(n_samples: int = 1000) -> list[str]:
    """Train encoder (if cache absent) and return *n_samples* SMILES.

    Sampling: draw latent vectors, find nearest encoded training molecule.
    """
    cache_path = config.GENERATION_RESULTS_DIR / "graph_latents.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        latents = data["z"]
        smiles_train = data["smiles"].tolist()
        LOGGER.info(f"Loaded cached graph latents ({len(smiles_train)} molecules)")
    else:
        LOGGER.info("Preparing graph dataset…")
        df = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
        dataset: list[Data] = []
        smiles_train: list[str] = []
        for smi in df["SMILES"].unique().to_list():  # type: ignore[no-any-return]
            g = mol_to_graph(smi)
            if g is not None and g.edge_index.size(1) > 0:
                dataset.append(g)
                smiles_train.append(smi)
        LOGGER.info(f"Graphs: {len(dataset)}")

        enc = train_encoder(dataset)
        enc.eval()
        latents_list = []
        loader = DataLoader(dataset, batch_size=128)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEV)
                mu, _ = enc(batch)
                latents_list.append(mu.cpu().numpy())
        latents = np.vstack(latents_list)
        np.savez_compressed(cache_path, z=latents, smiles=smiles_train)
        LOGGER.info(f"Latents cached to {cache_path}")

    # Build simple k-NN (cosine) index
    from sklearn.neighbors import NearestNeighbors  # type: ignore

    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(latents)

    rng = np.random.default_rng(config.RANDOM_STATE)
    sampled = []
    while len(sampled) < n_samples:
        z = rng.standard_normal(LATENT_DIM).reshape(1, -1)
        dist, idx = nn_model.kneighbors(z, return_distance=True)
        smi = smiles_train[int(idx[0][0])]
        sampled.append(smi)

    # Deduplicate while keeping order
    seen: set[str] = set()
    unique_smiles = []
    for smi in sampled:
        if smi not in seen:
            seen.add(smi)
            unique_smiles.append(smi)
        if len(unique_smiles) >= n_samples:
            break

    LOGGER.info(f"Graph generator produced {len(unique_smiles)} SMILES (unique) via nearest-neighbour lookup")
    # Save raw output
    out_path = config.GENERATION_RESULTS_DIR / "generated_smiles_raw.txt"
    Path(out_path).write_text("\n".join(unique_smiles))
    return unique_smiles
