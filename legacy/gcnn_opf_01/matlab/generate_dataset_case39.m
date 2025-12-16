clc; clear; close all;
baseMVA = 100.0;
k = 10;
n_train = 10000;
n_test = 2000;
n_unseen = 1200;
sigma_rel = 0.10;
target_penetration = 0.507;

data_dir = '../data_matlab';
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

wind_buses = [4, 7, 8, 15, 16, 18, 21, 25, 26, 28];
pv_buses   = [3, 12, 20, 23, 24, 27, 29, 39, 1, 9];

lambda_wind = 5.089; k_wind = 2.016;
v_cut_in = 4.0; v_rated = 12.0; v_cut_out = 25.0;
alpha_pv = 2.06; beta_pv = 2.5; G_STC = 1000.0;

fprintf('Loading case39...\n');
mpc = case39();
mpc.baseMVA = baseMVA; 

topo_seen = {
    [], 'Base'; 
    [6,7], 'Topo_1';
    [14,13], 'Topo_2';
    [2,3], 'Topo_3';
    [21,22], 'Topo_4'
};
topo_unseen = {
    [23,24], 'Unseen_1';
    [26,27], 'Unseen_2';
    [2,25], 'Unseen_3'
};

fprintf('Precomputing topology operators...\n');
[ops_seen, mpc_seen_list] = extract_topology_operators(mpc, topo_seen);
save(fullfile(data_dir, 'topology_operators.mat'), '-struct', 'ops_seen');

fprintf('Generating Training Set (%d)...\n', n_train);
[train_data, stats] = generate_samples(n_train, mpc, mpc_seen_list, wind_buses, pv_buses, lambda_wind, k_wind, v_cut_in, v_rated, v_cut_out, alpha_pv, beta_pv, G_STC, sigma_rel, target_penetration, k, ops_seen, true);
save(fullfile(data_dir, 'samples_train.mat'), '-struct', 'train_data');
save(fullfile(data_dir, 'norm_stats.mat'), '-struct', 'stats');

fprintf('Generating Test Set (%d)...\n', n_test);
[test_data, ~] = generate_samples(n_test, mpc, mpc_seen_list, wind_buses, pv_buses, lambda_wind, k_wind, v_cut_in, v_rated, v_cut_out, alpha_pv, beta_pv, G_STC, sigma_rel, target_penetration, k, ops_seen, false);
save(fullfile(data_dir, 'samples_test.mat'), '-struct', 'test_data');

fprintf('Generating Unseen Set (%d)...\n', n_unseen);
[ops_unseen, mpc_unseen_list] = extract_topology_operators(mpc, topo_unseen);
save(fullfile(data_dir, 'topology_operators_unseen.mat'), '-struct', 'ops_unseen');
[unseen_data, ~] = generate_samples(n_unseen, mpc, mpc_unseen_list, wind_buses, pv_buses, lambda_wind, k_wind, v_cut_in, v_rated, v_cut_out, alpha_pv, beta_pv, G_STC, sigma_rel, target_penetration, k, ops_unseen, false);
save(fullfile(data_dir, 'samples_unseen.mat'), '-struct', 'unseen_data');

fprintf('Done!\n');

function [ops, mpc_list] = extract_topology_operators(mpc_base, topo_defs)
    n_topo = size(topo_defs, 1);
    n_bus = size(mpc_base.bus, 1);
    ops.g_ndiag = zeros(n_topo, n_bus, n_bus);
    ops.b_ndiag = zeros(n_topo, n_bus, n_bus);
    ops.g_diag = zeros(n_topo, n_bus);
    ops.b_diag = zeros(n_topo, n_bus);
    ops.gen_bus_map = mpc_base.gen(:, 1); 
    ops.N_BUS = n_bus; ops.N_GEN = size(mpc_base.gen, 1);
    mpc_list = cell(n_topo, 1);
    for i = 1:n_topo
        pairs = topo_defs{i, 1};
        mpc_curr = apply_topology(mpc_base, pairs);
        mpc_list{i} = mpc_curr;
        [Ybus, ~, ~] = makeYbus(mpc_curr);
        G = real(full(Ybus)); B = imag(full(Ybus));
        ops.g_diag(i, :) = diag(G); ops.b_diag(i, :) = diag(B);
        G_nd = G - diag(diag(G)); B_nd = B - diag(diag(B));
        ops.g_ndiag(i, :, :) = G_nd; ops.b_ndiag(i, :, :) = B_nd;
    end
end
function [dataset, stats] = generate_samples(n_samples, mpc_base, mpc_topo_list, wind_buses, pv_buses, lam_w, k_w, v_in, v_rate, v_out, a_pv, b_pv, G_stc, sigma, pen_target, k_iter, ops, is_train)
    N_BUS = size(mpc_base.bus, 1); N_GEN = size(mpc_base.gen, 1);
    PD_base = mpc_base.bus(:, 3) / mpc_base.baseMVA; QD_base = mpc_base.bus(:, 4) / mpc_base.baseMVA;
    
    % Start parallel pool if not already started
    if isempty(gcp('nocreate')), parpool; end
    
    % Progress bar setup
    q = parallel.pool.DataQueue;
    afterEach(q, @updateProgress);
    fprintf('  0%% [');
    
    % Generate candidates in batches to handle failures
    % Heuristic: 1.2x samples to account for failures
    n_try = ceil(n_samples * 1.2);
    
    % Pre-allocate cell array for results
    results = cell(n_try, 1);
    
    % Parallel loop
    parfor i = 1:n_try
        % Suppress singular matrix warnings inside worker
        warning('off', 'MATLAB:nearlySingularMatrix');
        warning('off', 'MATLAB:singularMatrix');
        
        % Random topology
        tid = randi(length(mpc_topo_list));
        mpc_curr = mpc_topo_list{tid};
        
        % Random load/RES
        noise = randn(N_BUS, 1);
        pd_raw = max(PD_base .* (1 + sigma * noise), 0);
        
        % Wind
        v_wind = wblrnd(lam_w, k_w, [length(wind_buses), 1]);
        idx = (v_wind >= v_in) & (v_wind < v_rate); 
        cf_wind = zeros(size(v_wind)); 
        cf_wind(idx) = ((v_wind(idx) - v_in) / (v_rate - v_in)).^3; 
        cf_wind(v_wind >= v_rate & v_wind < v_out) = 1.0;
        p_wind = cf_wind .* PD_base(wind_buses);
        
        % PV
        irr_pv = betarnd(a_pv, b_pv, [length(pv_buses), 1]) * G_stc;
        cf_pv = min(irr_pv / G_stc, 1.0); cf_pv = max(cf_pv, 0.0);
        p_pv = cf_pv .* PD_base(pv_buses);
        
        % Penetration scaling
        total_load = sum(pd_raw); total_res_avail = sum(p_wind) + sum(p_pv);
        if total_res_avail > 0
            scale = (pen_target * total_load) / total_res_avail; 
            p_wind = p_wind * scale; 
            p_pv = p_pv * scale; 
        end
        
        pd_net = pd_raw; 
        pd_net(wind_buses) = pd_net(wind_buses) - p_wind; 
        pd_net(pv_buses) = pd_net(pv_buses) - p_pv;
        pf_ratio = QD_base ./ (PD_base + 1e-8); 
        qd_net = pf_ratio .* pd_net;
        
        % Feature Construction
        G_d = reshape(ops.g_diag(tid, :), [], 1); B_d = reshape(ops.b_diag(tid, :), [], 1);
        G_nd = squeeze(ops.g_ndiag(tid, :, :)); B_nd = squeeze(ops.b_ndiag(tid, :, :));
        G_full = G_nd + diag(G_d); B_full = B_nd + diag(B_d);
        
        gen_idx = mpc_curr.gen(:, 1);
        PMAX = mpc_curr.gen(:, 9) / mpc_base.baseMVA; PMIN = mpc_curr.gen(:, 10) / mpc_base.baseMVA;
        QMAX = mpc_curr.gen(:, 4) / mpc_base.baseMVA; QMIN = mpc_curr.gen(:, 5) / mpc_base.baseMVA;
        
        [e_k, f_k, converged] = construct_features(pd_net, qd_net, G_full, B_full, G_nd, B_nd, G_d, B_d, gen_idx, PMAX, PMIN, QMAX, QMIN, k_iter);
        
        if converged
            % Run OPF
            mpc_run = mpc_curr; 
            mpc_run.bus(:, 3) = pd_net * mpc_base.baseMVA; 
            mpc_run.bus(:, 4) = qd_net * mpc_base.baseMVA;
            opt = mpoption('verbose', 0, 'out.all', 0); 
            res = runopf(mpc_run, opt);
            
            if res.success == 1
                % Store result in struct
                s = struct();
                s.e_0_k = e_k;
                s.f_0_k = f_k;
                s.pd = pd_net';
                s.qd = qd_net';
                s.topo_id = tid - 1;
                s.pg_labels = (res.gen(:, 2) / mpc_base.baseMVA)';
                s.vg_labels = (res.bus(gen_idx, 8))';
                results{i} = s;
            else
                results{i} = [];
            end
        else
            results{i} = [];
        end
        send(q, 1);
    end
    
    fprintf('] Done.\n');
    
    % Filter valid results
    valid_results = results(~cellfun('isempty', results));
    n_valid = length(valid_results);
    fprintf('  Valid samples generated: %d / %d required\n', n_valid, n_samples);
    
    if n_valid < n_samples
        error('Not enough valid samples generated. Increase buffer or check solver settings.');
    end
    
    % Truncate
    valid_results = valid_results(1:n_samples);
    
    % Convert cell array of structs to struct array
    valid_structs = [valid_results{:}];

    % Convert to struct of arrays
    % e_0_k is [N_BUS, k]. We want [n_samples, N_BUS, k].
    % cat(3, ...) stacks along 3rd dim. Then permute.
    tmp_e = cat(3, valid_structs.e_0_k); % [N_BUS, k, n_samples]
    D.e_0_k = permute(tmp_e, [3, 1, 2]);
    
    tmp_f = cat(3, valid_structs.f_0_k);
    D.f_0_k = permute(tmp_f, [3, 1, 2]);
    
    D.pd = vertcat(valid_structs.pd);
    D.qd = vertcat(valid_structs.qd);
    D.topo_id = vertcat(valid_structs.topo_id);
    D.pg_labels = vertcat(valid_structs.pg_labels);
    D.vg_labels = vertcat(valid_structs.vg_labels);
    
    dataset = D; 
    stats = struct();
    if is_train
        stats.pd_mean = mean(D.pd(:)); stats.pd_std = std(D.pd(:)); 
        stats.qd_mean = mean(D.qd(:)); stats.qd_std = std(D.qd(:)); 
        stats.pg_mean = mean(D.pg_labels(:)); stats.pg_std = std(D.pg_labels(:)); 
        stats.vg_mean = mean(D.vg_labels(:)); stats.vg_std = std(D.vg_labels(:)); 
    end
    
    function updateProgress(~)
        persistent p
        if isempty(p), p = 0; end
        p = p + 1;
        if mod(p, floor(n_try/50)) == 0
            fprintf('.');
        end
    end
end
