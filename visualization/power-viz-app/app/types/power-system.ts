// Type definitions for PYPOWER case data

export interface Bus {
    id: number;
    index: number;
    type: number;
    type_name: 'PQ' | 'PV' | 'Slack' | 'Isolated' | 'Unknown';
    pd: number;  // MW
    qd: number;  // MVAr
    vm: number;  // p.u.
    va: number;  // degrees
    baseKV: number;
    vmax: number;
    vmin: number;
    area: number;
}

export interface Branch {
    id: number;
    from_bus: number;
    to_bus: number;
    resistance: number;  // p.u.
    reactance: number;   // p.u.
    charging: number;    // p.u.
    rateA: number;       // MVA
    rateB: number;       // MVA
    rateC: number;       // MVA
    tap: number;
    shift: number;       // degrees
    status: number;      // 1=in-service, 0=out
}

export interface Generator {
    id: number;
    bus: number;
    pg: number;    // MW
    qg: number;    // MVAr
    qmax: number;  // MVAr
    qmin: number;  // MVAr
    vg: number;    // p.u.
    pmax: number;  // MW
    pmin: number;  // MW
    status: number;
}

export interface CaseData {
    baseMVA: number;
    name: string;
    buses: Bus[];
    branches: Branch[];
    generators: Generator[];
}

export interface NetworkNode extends Bus {
    x?: number;
    y?: number;
    vx?: number;
    vy?: number;
    fx?: number | null;
    fy?: number | null;
    hasGenerator?: boolean;
    island?: number;
}

export interface NetworkLink extends Branch {
    source: NetworkNode | number;
    target: NetworkNode | number;
    index?: number;
    active?: boolean;
}

export interface ConnectivityResult {
    isConnected: boolean;
    numIslands: number;
    islands: number[][];
}
