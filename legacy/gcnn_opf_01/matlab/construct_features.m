function [e_0_k, f_0_k, converged] = construct_features(pd, qd, G, B, g_ndiag, b_ndiag, g_diag, b_diag, gen_bus_indices, PMAX, PMIN, QMAX, QMIN, k_iter, eps_val)
    if nargin < 15, eps_val = 1e-8; end
    N_BUS = length(pd);
    e = ones(N_BUS, 1); f = zeros(N_BUS, 1);
    e_0_k = zeros(N_BUS, k_iter); f_0_k = zeros(N_BUS, k_iter);
    converged = true;
    
    for iter = 1:k_iter
        Ge = G * e; Gf = G * f; Be = B * e; Bf = B * f;
        PG_bus = pd + e .* Ge - e .* Bf + f .* Gf + f .* Be;
        QG_bus = qd - f .* Ge + f .* Bf + e .* Gf - e .* Be;
        
        PG_gen = PG_bus(gen_bus_indices); QG_gen = QG_bus(gen_bus_indices);
        PG_clamped = max(min(PG_gen, PMAX), PMIN);
        QG_clamped = max(min(QG_gen, QMAX), QMIN);
        PG_bus(gen_bus_indices) = PG_clamped; QG_bus(gen_bus_indices) = QG_clamped;
        
        pd_eff = PG_bus - (e .* Ge - e .* Bf + f .* Gf + f .* Be);
        qd_eff = QG_bus - (e .* Gf - e .* Be - f .* Ge + f .* Bf);
        
        alpha = g_ndiag * e - b_ndiag * f;
        beta  = g_ndiag * f + b_ndiag * e;
        s = e.^2 + f.^2;
        
        delta = -pd_eff - s .* g_diag;
        lambda_val = -qd_eff - s .* b_diag;
        
        denom = alpha.^2 + beta.^2 + eps_val;
        e_next = (delta .* alpha - lambda_val .* beta) ./ denom;
        f_next = (delta .* beta + lambda_val .* alpha) ./ denom;
        
        v_mag = sqrt(e_next.^2 + f_next.^2 + eps_val);
        e_next = e_next ./ v_mag; f_next = f_next ./ v_mag;
        
        e_0_k(:, iter) = e_next; f_0_k(:, iter) = f_next;
        e = e_next; f = f_next;
    end
    if any(isnan(e)) || any(isinf(e)), converged = false; end
end
