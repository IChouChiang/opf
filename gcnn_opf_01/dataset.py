"""PyTorch Dataset for GCNN OPF training."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class OPFDataset(Dataset):
    """
    Dataset for GCNN OPF training/testing.

    Loads precomputed samples from NPZ files containing:
    - e_0_k, f_0_k: [N_SAMPLES, N_BUS, k] feature construction outputs
    - pd, qd: [N_SAMPLES, N_BUS] demand scenarios
    - topo_id: [N_SAMPLES] topology IDs (0-4 for case6ww)
    - pg_labels, vg_labels: [N_SAMPLES, N_GEN] OPF solution labels

    Also loads topology-specific operators (G, B matrices) for physics loss.
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

        # Features: [N, N_BUS, k]
        self.e_0_k = torch.from_numpy(data["e_0_k"]).float()
        self.f_0_k = torch.from_numpy(data["f_0_k"]).float()

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

        # Operators are saved as [N_TOPOLOGIES, ...] arrays
        # Convert to dict: topo_id -> {g_ndiag, b_ndiag, g_diag, b_diag}
        self.topology_operators = {}
        n_topologies = topo_data["g_ndiag"].shape[0]

        for topo_id in range(n_topologies):
            self.topology_operators[topo_id] = {
                "g_ndiag": torch.from_numpy(topo_data["g_ndiag"][topo_id]).float(),
                "b_ndiag": torch.from_numpy(topo_data["b_ndiag"][topo_id]).float(),
                "g_diag": torch.from_numpy(topo_data["g_diag"][topo_id]).float(),
                "b_diag": torch.from_numpy(topo_data["b_diag"][topo_id]).float(),
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
                'e_0_k': [N_BUS, k]
                'f_0_k': [N_BUS, k]
                'pd': [N_BUS]
                'qd': [N_BUS]
                'topo_id': int
                'operators': dict of {G, B, g_diag, b_diag, g_ndiag, b_ndiag} [N_BUS, N_BUS] or [N_BUS]
                'pg_label': [N_GEN]
                'vg_label': [N_GEN]
                'gen_label': [N_GEN, 2] stacked (PG, VG)
        """
        topo_id = int(self.topo_id[idx])

        return {
            "e_0_k": self.e_0_k[idx],  # [N_BUS, k]
            "f_0_k": self.f_0_k[idx],  # [N_BUS, k]
            "pd": self.pd[idx],  # [N_BUS]
            "qd": self.qd[idx],  # [N_BUS]
            "topo_id": topo_id,
            "operators": self.topology_operators[topo_id],  # dict of tensors
            "pg_label": self.pg_labels[idx],  # [N_GEN]
            "vg_label": self.vg_labels[idx],  # [N_GEN]
            "gen_label": torch.stack(
                [self.pg_labels[idx], self.vg_labels[idx]], dim=-1
            ),  # [N_GEN, 2]
        }

    def get_norm_stats(self):
        """Return normalization statistics for denormalization."""
        return self.norm_stats
