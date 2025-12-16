function mpc_topo = apply_topology(mpc_base, branch_pairs)
    mpc_topo = mpc_base;
    if isempty(branch_pairs), return; end
    F_BUS = 1; T_BUS = 2; BR_STATUS = 11;
    for i = 1:size(branch_pairs, 1)
        f = branch_pairs(i, 1); t = branch_pairs(i, 2);
        mask = (mpc_topo.branch(:, F_BUS) == f & mpc_topo.branch(:, T_BUS) == t) | (mpc_topo.branch(:, F_BUS) == t & mpc_topo.branch(:, T_BUS) == f);
        if sum(mask) > 0, mpc_topo.branch(mask, BR_STATUS) = 0; end
    end
end
