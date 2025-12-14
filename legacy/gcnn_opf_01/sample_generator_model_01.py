# sample_generator_model_01.py

import numpy as np


def wind_power_curve(v, P_rated=1.0, v_cut_in=4.0, v_rated=12.0, v_cut_out=25.0):
    """
    v : wind speed (m/s) at bus
    Returns: capacity factor in [0,1]
    """
    if v < v_cut_in:
        return 0.0
    elif v_cut_in <= v < v_rated:
        return ((v - v_cut_in) / (v_rated - v_cut_in)) ** 3
    elif v_rated <= v < v_cut_out:
        return 1.0
    else:
        return 0.0


def pv_power_curve(s, P_rated=1.0, G_stc=1000.0):
    """
    s : irradiance (W/m²) at bus
    Returns: capacity factor in [0,1]
    """
    cf = s / G_stc
    if cf > 1.0:
        cf = 1.0
    if cf < 0.0:
        cf = 0.0
    return cf


class SampleGeneratorModel01:
    """
    Scenario generator for model_01.

    Responsibilities:
        - Start from base PD/QD (case39).
        - Add Gaussian load fluctuation.
        - Sample RES (wind, PV), map to power.
        - Scale RES to match target penetration.
        - Treat RES as negative demand -> final pd, qd.

    This class does NOT touch topology directly; it just takes
    a topology_id so it can be carried along in the sample dict.
    """

    def __init__(
        self,
        PD_base: np.ndarray,  # [N_BUS]
        QD_base: np.ndarray,  # [N_BUS]
        penetration_target: float = 0.507,  # default 50.7% RES penetration (interpreted as fraction)
        res_bus_idx_wind=None,  # list[int] bus indices (0-based) with wind
        res_bus_idx_pv=None,  # list[int] bus indices (0-based) with PV
        rng_seed: int = 0,
        sigma_rel: float = 0.1,
        lam_wind: float = 5.089,
        k_wind: float = 2.016,
        v_cut_in: float = 4.0,
        v_rated: float = 12.0,
        v_cut_out: float = 25.0,
        alpha_pv: float = 2.06,
        beta_pv: float = 2.5,
        g_stc: float = 1000.0,
        allow_negative_pd: bool = False,  # if True, do NOT clip resulting pd >= 0
    ):
        self.PD_base = PD_base.copy()
        self.QD_base = QD_base.copy()
        self.N_BUS = PD_base.shape[0]

        # Accept either fraction (e.g., 0.507) or percent (e.g., 50.7)
        pt = float(penetration_target)
        if pt > 1.0:  # assume user passed percent
            pt = pt / 100.0
        self.penetration_target = pt
        self.allow_negative_pd = bool(allow_negative_pd)

        self.rng = np.random.default_rng(rng_seed)

        # RES parameters from sample_config_model_01.py
        self.sigma_rel = sigma_rel
        self.lam_wind = lam_wind
        self.k_wind = k_wind
        self.v_cut_in = v_cut_in
        self.v_rated = v_rated
        self.v_cut_out = v_cut_out
        self.alpha_pv = alpha_pv
        self.beta_pv = beta_pv
        self.g_stc = g_stc

        # RES bus sets (0-based internal indices)
        # Get from sample_config_model_01.get_res_bus_indices(ppc_int)
        if res_bus_idx_wind is None:
            res_bus_idx_wind = []
        if res_bus_idx_pv is None:
            res_bus_idx_pv = []

        self.res_bus_idx_wind = np.array(res_bus_idx_wind, dtype=int)
        self.res_bus_idx_pv = np.array(res_bus_idx_pv, dtype=int)

        # Prebuild masks for convenience
        self.res_mask_wind = np.zeros(self.N_BUS, dtype=bool)
        self.res_mask_pv = np.zeros(self.N_BUS, dtype=bool)
        self.res_mask_wind[self.res_bus_idx_wind] = True
        self.res_mask_pv[self.res_bus_idx_pv] = True

    def sample_scenario(self, topology_id: int):
        """
        Generate a single scenario for the given topology_id.

        Returns:
            dict with keys:
                "pd"           : np.ndarray [N_BUS]  (after RES, fluctuated)
                "qd"           : np.ndarray [N_BUS]
                "topology_id"  : int
                "P_res_avail"  : np.ndarray [N_BUS] (optional: raw available RES power)
                "pd_raw"       : np.ndarray [N_BUS] (before RES, after fluctuation)
                "qd_raw"       : np.ndarray [N_BUS]
        """
        # ----- 1) Load fluctuation around base PD/QD -----
        noise = self.rng.standard_normal(size=self.N_BUS)
        pd_raw = self.PD_base * (1.0 + self.sigma_rel * noise)
        if not self.allow_negative_pd:
            pd_raw = np.clip(pd_raw, 0.0, None)

        # Calculate original power factor (QD/PD ratio) from base case
        # This will be applied to final adjusted PD to get consistent QD
        power_factor_ratio = self.QD_base / (self.PD_base + 1e-8)
        qd_raw = power_factor_ratio * pd_raw

        # ----- 2) RES sampling: wind + PV → available power -----
        # Initialize zero available power
        P_res_avail = np.zeros(self.N_BUS, dtype=float)

        # 2.1 wind (Weibull)
        if self.res_bus_idx_wind.size > 0:
            # Draw wind speed for each wind bus using Weibull distribution
            v = (
                self.rng.weibull(self.k_wind, size=self.res_bus_idx_wind.size)
                * self.lam_wind
            )

            # Apply wind power curve to get capacity factors [0,1]
            P_wind_cf = np.array(
                [
                    wind_power_curve(
                        vi, 1.0, self.v_cut_in, self.v_rated, self.v_cut_out
                    )
                    for vi in v
                ]
            )

            # Nameplate capacity limited by local base load (avoid negative net load)
            cap_wind = self.PD_base[self.res_bus_idx_wind]
            P_wind_avail = P_wind_cf * cap_wind

            P_res_avail[self.res_bus_idx_wind] += P_wind_avail

        # 2.2 PV (Beta)
        if self.res_bus_idx_pv.size > 0:
            # Draw irradiance from Beta distribution, scale to W/m²
            s_norm = self.rng.beta(
                self.alpha_pv, self.beta_pv, size=self.res_bus_idx_pv.size
            )
            s = s_norm * self.g_stc  # irradiance in W/m²

            # Apply PV power curve to get capacity factors [0,1]
            P_pv_cf = np.array([pv_power_curve(si, 1.0, self.g_stc) for si in s])

            # Nameplate capacity limited by local base load (avoid negative net load)
            cap_pv = self.PD_base[self.res_bus_idx_pv]
            P_pv_avail = P_pv_cf * cap_pv

            P_res_avail[self.res_bus_idx_pv] += P_pv_avail

        # ----- 3) Enforce target penetration -----
        P_total_load_raw = pd_raw.sum()
        P_res_sum = P_res_avail.sum()

        if P_res_sum > 0.0:
            scale = self.penetration_target * P_total_load_raw / P_res_sum
            P_res_injected = P_res_avail * scale
        else:
            P_res_injected = P_res_avail.copy()  # all zero

        # ----- 4) Treat RES as negative loads -----
        pd = pd_raw - P_res_injected
        if not self.allow_negative_pd:
            pd = np.clip(pd, 0.0, None)  # enforce non-negative if flag False

        # Recalculate QD based on final PD to maintain consistent power factor
        qd = power_factor_ratio * pd

        # ----- 5) Return scenario dict -----
        return {
            "pd": pd,
            "qd": qd,
            "topology_id": int(topology_id),
            "P_res_avail": P_res_avail,
            "pd_raw": pd_raw,
            "qd_raw": qd_raw,
        }
