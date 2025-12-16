"""Unified dataset for OPF model training and evaluation.

Supports two feature types:
- 'flat': Concatenated vector [pd, qd, g_flat, b_flat] for DNN models
- 'graph': Node-wise features (e_0_k, f_0_k) for GCNN models
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class OPFDataset(Dataset):
    """
    Unified Dataset for OPF model training/evaluation.

    Precomputed samples from NPZ files always containing:
    - pd, qd: [N_SAMPLES, N_BUS] demand scenarios
    - topo_id: [N_SAMPLES] topology IDs (0-4 for case6ww)
    - pg_labels, vg_labels: [N_SAMPLES, N_GEN] OPF solution labels

    Conditionally loads (for 'graph' feature_type):
    - e_0_k, f_0_k: [N_SAMPLES, N_BUS, k] feature construction outputs

    Also loads topology-specific operators (G, B matrices) for physics loss.

    Feature Types:
    - 'flat': Returns concatenated [pd, qd, g_flat, b_flat] vector (for DNN)
    - 'graph': Returns e_0_k, f_0_k node features (for GCNN)
    """

    def __init__(
        self,
        data_path: str | Path,
        topo_path: str | Path,
        norm_path: str | Path | None = None,
        normalize: bool = True,
        split: str = "train",
        feature_type: str = "flat",
        feature_params: dict | None = None,
    ):
        """
        Args:
            data_path: Path to samples_{split}.npz file
            topo_path: Path to topology_operators.npz
            norm_path: Path to norm_stats.npz (optional)
            normalize: Whether to apply z-score normalization
            split: 'train' or 'eval'
            feature_type: Type of input features ('flat' or 'graph')
            feature_params: Additional parameters for feature processing
                - For 'graph': {'feature_iterations': int} to slice e_0_k, f_0_k
        """
        self.split = split
        self.normalize = normalize
        self.feature_type = feature_type
        self.feature_params = feature_params or {}
        self.data_path: Path = Path(data_path)

        if feature_type not in ("flat", "graph"):
            raise ValueError(
                f"feature_type must be 'flat' or 'graph', got {feature_type}"
            )

        # Load main dataset
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        data = np.load(self.data_path)

        # Demands: [N, N_BUS]
        self.pd = torch.from_numpy(data["pd"]).float()
        self.qd = torch.from_numpy(data["qd"]).float()

        # Topology IDs: [N]
        self.topo_id = torch.from_numpy(data["topo_id"]).long()

        # Labels: [N, N_GEN]
        self.pg_labels = torch.from_numpy(data["pg_labels"]).float()
        self.vg_labels = torch.from_numpy(data["vg_labels"]).float()

        self.n_samples = len(self.topo_id)

        # Load graph features if needed
        self.e_0_k: torch.Tensor | None = None
        self.f_0_k: torch.Tensor | None = None

        if feature_type == "graph":
            if "e_0_k" not in data or "f_0_k" not in data:
                raise ValueError(
                    "Dataset does not contain e_0_k/f_0_k features required for 'graph' feature_type"
                )
            self.e_0_k = torch.from_numpy(data["e_0_k"]).float()
            self.f_0_k = torch.from_numpy(data["f_0_k"]).float()

            # Slice features if feature_iterations is specified
            feature_iterations = self.feature_params.get("feature_iterations")
            if feature_iterations is not None:
                k_avail = self.e_0_k.shape[2]
                if feature_iterations > k_avail:
                    raise ValueError(
                        f"Requested {feature_iterations} iterations but dataset only has {k_avail}"
                    )
                self.e_0_k = self.e_0_k[:, :, :feature_iterations]
                self.f_0_k = self.f_0_k[:, :, :feature_iterations]

        # Load topology operators (G, B matrices for each topology)
        self.topo_path: Path = Path(topo_path)
        if not self.topo_path.exists():
            raise FileNotFoundError(f"Topology operators not found: {topo_path}")

        topo_data = np.load(self.topo_path)

        self.topology_operators: dict[int, dict[str, torch.Tensor]] = {}
        self.topology_vectors: dict[int, dict[str, torch.Tensor]] = {}
        n_topologies = topo_data["g_ndiag"].shape[0]

        for topo_id in range(n_topologies):
            g_ndiag = torch.from_numpy(topo_data["g_ndiag"][topo_id]).float()
            b_ndiag = torch.from_numpy(topo_data["b_ndiag"][topo_id]).float()
            g_diag = torch.from_numpy(topo_data["g_diag"][topo_id]).float()
            b_diag = torch.from_numpy(topo_data["b_diag"][topo_id]).float()

            # Store operators for physics loss
            self.topology_operators[topo_id] = {
                "g_ndiag": g_ndiag,
                "b_ndiag": b_ndiag,
                "g_diag": g_diag,
                "b_diag": b_diag,
            }

            # Precompute flattened vectors for 'flat' feature type
            if feature_type == "flat":
                G_full = g_ndiag + torch.diag(g_diag)
                B_full = b_ndiag + torch.diag(b_diag)
                self.topology_vectors[topo_id] = {
                    "g_flat": G_full.reshape(-1),
                    "b_flat": B_full.reshape(-1),
                }

        # Store generator bus mapping
        self.gen_bus_map = torch.from_numpy(topo_data["gen_bus_map"]).long()

        # Load and apply normalization stats
        self.norm_stats: dict[str, torch.Tensor] | None = None
        if normalize and norm_path:
            norm_path_obj = Path(norm_path)
            if norm_path_obj.exists():
                stats = np.load(norm_path_obj)
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

                # Apply z-score normalization
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

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a sample dict based on feature_type.

        For 'flat':
            'input': [2*N_BUS + 2*N_BUS^2] concatenated vector [pd, qd, g_flat, b_flat]
            'pd': [N_BUS]
            'qd': [N_BUS]
            'topo_id': int
            'operators': dict of {g_ndiag, b_ndiag, g_diag, b_diag}
            'pg_label': [N_GEN]
            'vg_label': [N_GEN]
            'gen_label': [N_GEN, 2] stacked (PG, VG)

        For 'graph':
            'e_0_k': [N_BUS, k]
            'f_0_k': [N_BUS, k]
            'pd': [N_BUS]
            'qd': [N_BUS]
            'topo_id': int
            'operators': dict of {g_ndiag, b_ndiag, g_diag, b_diag}
            'pg_label': [N_GEN]
            'vg_label': [N_GEN]
            'gen_label': [N_GEN, 2] stacked (PG, VG)
        """
        topo_id = int(self.topo_id[idx])
        pd = self.pd[idx]
        qd = self.qd[idx]

        # Common fields
        sample = {
            "pd": pd,
            "qd": qd,
            "topo_id": topo_id,
            "operators": self.topology_operators[topo_id],
            "pg_label": self.pg_labels[idx],
            "vg_label": self.vg_labels[idx],
            "gen_label": torch.stack(
                [self.pg_labels[idx], self.vg_labels[idx]], dim=-1
            ),
        }

        if self.feature_type == "flat":
            # Construct input vector: [pd, qd, g_flat, b_flat]
            g_flat = self.topology_vectors[topo_id]["g_flat"]
            b_flat = self.topology_vectors[topo_id]["b_flat"]
            sample["input"] = torch.cat([pd, qd, g_flat, b_flat], dim=0)

        elif self.feature_type == "graph":
            # Node-wise features for GCNN
            assert self.e_0_k is not None and self.f_0_k is not None
            sample["e_0_k"] = self.e_0_k[idx]
            sample["f_0_k"] = self.f_0_k[idx]

        return sample

    def get_norm_stats(self) -> dict[str, torch.Tensor] | None:
        """Return normalization statistics for denormalization."""
        return self.norm_stats

    def get_gen_bus_map(self) -> torch.Tensor:
        """Return generator-to-bus mapping."""
        return self.gen_bus_map
