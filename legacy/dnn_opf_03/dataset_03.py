"""PyTorch Dataset for DeepOPF-FT Baseline (DNN)."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class OPFDataset03(Dataset):
    """
    Dataset for DeepOPF-FT Baseline (DNN).

    Input:
        x = [pd, qd, vec(G), vec(B)]
        Dimension: 2*N + 2*N^2

    Loads precomputed samples from NPZ files.
    """

    def __init__(
        self,
        data_path,
        topo_operators_path,
        norm_stats_path=None,
        normalize: bool = True,
        split: str = "train",
    ):
        """
        Args:
            data_path: Path to samples_{split}.npz file
            topo_operators_path: Path to topology_operators.npz
            norm_stats_path: Path to norm_stats.npz (optional)
            normalize: Whether to apply z-score normalization
            split: 'train' or 'test'
        """
        self.split = split
        self.normalize = normalize

        # Load main dataset
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        data = np.load(data_path)

        # Demands: [N, N_BUS]
        self.pd = torch.from_numpy(data["pd"]).float()
        self.qd = torch.from_numpy(data["qd"]).float()

        # Topology IDs: [N]
        self.topo_id = torch.from_numpy(data["topo_id"]).long()

        # Labels: [N, N_GEN]
        self.pg_labels = torch.from_numpy(data["pg_labels"]).float()
        self.vg_labels = torch.from_numpy(data["vg_labels"]).float()

        self.n_samples = len(self.topo_id)

        # Load topology operators (G, B matrices for each topology)
        topo_path = Path(topo_operators_path)
        if not topo_path.exists():
            raise FileNotFoundError(f"Topology operators not found: {topo_path}")

        topo_data = np.load(topo_path)

        # Precompute flattened G and B vectors for each topology
        # G = G_ndiag + diag(G_diag)
        # B = B_ndiag + diag(B_diag)
        self.topology_operators = {}
        self.topology_vectors = {}  # Store flattened vectors

        n_topologies = topo_data["g_ndiag"].shape[0]

        for topo_id in range(n_topologies):
            g_ndiag = torch.from_numpy(topo_data["g_ndiag"][topo_id]).float()
            b_ndiag = torch.from_numpy(topo_data["b_ndiag"][topo_id]).float()
            g_diag = torch.from_numpy(topo_data["g_diag"][topo_id]).float()
            b_diag = torch.from_numpy(topo_data["b_diag"][topo_id]).float()

            # Reconstruct full matrices
            G_full = g_ndiag + torch.diag(g_diag)
            B_full = b_ndiag + torch.diag(b_diag)

            # Flatten
            g_flat = G_full.reshape(-1)  # [N*N]
            b_flat = B_full.reshape(-1)  # [N*N]

            self.topology_vectors[topo_id] = {"g_flat": g_flat, "b_flat": b_flat}

            # Keep operators for physics loss
            self.topology_operators[topo_id] = {
                "g_ndiag": g_ndiag,
                "b_ndiag": b_ndiag,
                "g_diag": g_diag,
                "b_diag": b_diag,
            }

        # Store generator bus mapping
        self.gen_bus_map = torch.from_numpy(topo_data["gen_bus_map"]).long()

        # Load normalization stats
        self.norm_stats = None
        if normalize and norm_stats_path:
            norm_path = Path(norm_stats_path)
            if norm_path.exists():
                stats = np.load(norm_path)
                self.norm_stats = {
                    "pd_mean": torch.tensor(stats["pd_mean"]).float(),
                    "pd_std": torch.tensor(stats["pd_std"]).float(),
                    "qd_mean": torch.tensor(stats["qd_mean"]).float(),
                    "qd_std": torch.tensor(stats["qd_std"]).float(),
                    "pg_mean": torch.tensor(stats["pg_mean"]).float(),
                    "pg_std": torch.tensor(stats["pg_std"]).float(),
                    "vg_mean": torch.tensor(stats["vg_mean"]).float(),
                    "vg_std": torch.tensor(stats["vg_std"]).float(),
                }

                # Apply normalization to loaded data
                self.pd = (self.pd - self.norm_stats["pd_mean"]) / (
                    self.norm_stats["pd_std"] + 1e-8
                )
                self.qd = (self.qd - self.norm_stats["qd_mean"]) / (
                    self.norm_stats["qd_std"] + 1e-8
                )
                self.pg_labels = (self.pg_labels - self.norm_stats["pg_mean"]) / (
                    self.norm_stats["pg_std"] + 1e-8
                )
                self.vg_labels = (self.vg_labels - self.norm_stats["vg_mean"]) / (
                    self.norm_stats["vg_std"] + 1e-8
                )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                'input': [2*N + 2*N^2] concatenated vector
                'topo_id': int
                'operators': dict of {G, B, g_diag, b_diag, g_ndiag, b_ndiag}
                'pg_label': [N_GEN]
                'vg_label': [N_GEN]
                'gen_label': [N_GEN, 2] stacked (PG, VG)
        """
        topo_id = int(self.topo_id[idx])

        # Construct input vector
        # x = [pd, qd, g_flat, b_flat]
        pd = self.pd[idx]
        qd = self.qd[idx]
        g_flat = self.topology_vectors[topo_id]["g_flat"]
        b_flat = self.topology_vectors[topo_id]["b_flat"]

        input_vec = torch.cat([pd, qd, g_flat, b_flat], dim=0)

        return {
            "input": input_vec,
            "pd": pd,  # Needed for physics loss
            "qd": qd,  # Needed for physics loss
            "topo_id": topo_id,
            "operators": self.topology_operators[topo_id],
            "pg_label": self.pg_labels[idx],
            "vg_label": self.vg_labels[idx],
            "gen_label": torch.stack(
                [self.pg_labels[idx], self.vg_labels[idx]], dim=-1
            ),
        }

    def get_norm_stats(self):
        """Return normalization statistics for denormalization."""
        return self.norm_stats
