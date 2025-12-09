'use client';

import React, { useState, useEffect, useMemo } from 'react';
import NetworkGraph from './components/NetworkGraph';
import { CaseData, NetworkNode, NetworkLink } from './types/power-system';
import { analyzeConnectivity, assignIslands } from './utils/graphAnalysis';

export default function Home() {
  const [caseData, setCaseData] = useState<CaseData | null>(null);
  const [activeBranches, setActiveBranches] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);
  const [showBranchPanel, setShowBranchPanel] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetch('/case39_data.json')
      .then(res => res.json())
      .then((data: CaseData) => {
        setCaseData(data);
        setActiveBranches(new Set(data.branches.map(b => b.id)));
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load case data:', err);
        setLoading(false);
      });
  }, []);

  const { nodes, links } = useMemo(() => {
    if (!caseData) return { nodes: [], links: [] };

    const genBuses = new Set(caseData.generators.map(g => g.bus));
    const nodes: NetworkNode[] = caseData.buses.map(bus => ({
      ...bus,
      hasGenerator: genBuses.has(bus.id),
    }));

    const links: NetworkLink[] = caseData.branches.map(branch => ({
      ...branch,
      source: branch.from_bus,
      target: branch.to_bus,
      active: activeBranches.has(branch.id),
    }));

    return { nodes, links };
  }, [caseData, activeBranches]);

  const connectivity = useMemo(() => {
    if (nodes.length === 0) {
      return { isConnected: true, numIslands: 1, islands: [] };
    }
    const result = analyzeConnectivity(nodes, links);
    assignIslands(nodes, result.islands);
    return result;
  }, [nodes, links]);

  const filteredBranches = useMemo(() => {
    if (!caseData) return [];
    return caseData.branches.filter(branch => {
      if (!searchTerm) return true;
      return (
        branch.id.toString().includes(searchTerm) ||
        branch.from_bus.toString().includes(searchTerm) ||
        branch.to_bus.toString().includes(searchTerm)
      );
    });
  }, [caseData, searchTerm]);

  const handleToggleBranch = (branchId: number) => {
    setActiveBranches(prev => {
      const newSet = new Set(prev);
      if (newSet.has(branchId)) {
        newSet.delete(branchId);
      } else {
        newSet.add(branchId);
      }
      return newSet;
    });
  };

  const handleShowAll = () => {
    if (caseData) {
      setActiveBranches(new Set(caseData.branches.map(b => b.id)));
    }
  };

  const handleHideAll = () => {
    setActiveBranches(new Set());
  };

  if (loading) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0f172a', color: '#94a3b8' }}>
        Loading...
      </div>
    );
  }

  if (!caseData) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0f172a', color: '#f87171' }}>
        Failed to load case data
      </div>
    );
  }

  const activeCount = activeBranches.size;
  const totalBranches = caseData.branches.length;

  return (
    <div style={{ height: '100vh', background: '#0f172a', position: 'relative', overflow: 'hidden' }}>
      {/* Graph - Background layer */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 1 }}>
        <NetworkGraph nodes={nodes} links={links} />
      </div>

      {/* Top Left - Case Info */}
      <div style={{
        position: 'fixed',
        top: 16,
        left: 16,
        zIndex: 100,
        background: 'rgba(15, 23, 42, 0.9)',
        padding: '12px 16px',
        borderRadius: 8,
        border: '1px solid #334155',
        color: 'white'
      }}>
        <div style={{ fontWeight: 600, fontSize: 16 }}>{caseData.name}</div>
        <div style={{ fontSize: 12, color: '#94a3b8' }}>
          {caseData.buses.length} buses • {caseData.generators.length} generators
        </div>
      </div>

      {/* Top Right - Connectivity */}
      <div style={{
        position: 'fixed',
        top: 16,
        right: 16,
        zIndex: 100,
        background: connectivity.isConnected ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
        padding: '8px 16px',
        borderRadius: 8,
        border: `1px solid ${connectivity.isConnected ? '#10b981' : '#ef4444'}`,
        color: connectivity.isConnected ? '#6ee7b7' : '#fca5a5',
        display: 'flex',
        alignItems: 'center',
        gap: 8
      }}>
        <div style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: connectivity.isConnected ? '#10b981' : '#ef4444'
        }} />
        {connectivity.isConnected ? 'Connected' : `${connectivity.numIslands} Islands`}
      </div>

      {/* Bottom Left - Branches Button */}
      <button
        onClick={() => setShowBranchPanel(!showBranchPanel)}
        style={{
          position: 'fixed',
          bottom: showBranchPanel ? 'calc(40% + 16px)' : 16,
          left: 16,
          zIndex: 100,
          background: showBranchPanel ? '#2563eb' : 'rgba(15, 23, 42, 0.9)',
          padding: '10px 16px',
          borderRadius: 8,
          border: `1px solid ${showBranchPanel ? '#3b82f6' : '#334155'}`,
          color: 'white',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: 14,
          fontWeight: 500,
          transition: 'all 0.2s'
        }}
      >
        ⚡ Branches
        <span style={{
          background: activeCount < totalBranches ? 'rgba(234, 179, 8, 0.3)' : '#475569',
          padding: '2px 8px',
          borderRadius: 4,
          fontSize: 12,
          color: activeCount < totalBranches ? '#fde047' : '#94a3b8'
        }}>
          {activeCount}/{totalBranches}
        </span>
      </button>

      {/* Bottom Right - Legend */}
      <div style={{
        position: 'fixed',
        bottom: 16,
        right: 16,
        zIndex: 100,
        background: 'rgba(15, 23, 42, 0.9)',
        padding: '10px 14px',
        borderRadius: 8,
        border: '1px solid #334155',
        display: 'flex',
        gap: 16,
        fontSize: 12,
        color: '#cbd5e1'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#60a5fa' }} />
          PQ
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 10, height: 10, transform: 'rotate(45deg)', background: '#34d399' }} />
          PV
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ color: '#f87171' }}>★</span>
          Slack
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#fbbf24' }} />
          Gen
        </div>
      </div>

      {/* Branch Panel */}
      {showBranchPanel && (
        <div style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          height: '40%',
          zIndex: 99,
          background: 'rgba(15, 23, 42, 0.98)',
          borderTop: '1px solid #334155',
          padding: 16,
          display: 'flex',
          flexDirection: 'column'
        }}>
          {/* Panel Header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 12 }}>
            <span style={{ fontWeight: 600, color: 'white' }}>Branch Control</span>

            <input
              type="text"
              placeholder="Search bus..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              style={{
                flex: 1,
                maxWidth: 200,
                padding: '6px 12px',
                background: '#1e293b',
                border: '1px solid #475569',
                borderRadius: 6,
                color: 'white',
                fontSize: 13,
                outline: 'none'
              }}
            />

            <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
              <button
                onClick={handleShowAll}
                style={{
                  padding: '6px 12px',
                  background: 'rgba(16, 185, 129, 0.2)',
                  border: '1px solid #10b981',
                  borderRadius: 6,
                  color: '#6ee7b7',
                  cursor: 'pointer',
                  fontSize: 12
                }}
              >
                Enable All
              </button>
              <button
                onClick={handleHideAll}
                style={{
                  padding: '6px 12px',
                  background: 'rgba(239, 68, 68, 0.2)',
                  border: '1px solid #ef4444',
                  borderRadius: 6,
                  color: '#fca5a5',
                  cursor: 'pointer',
                  fontSize: 12
                }}
              >
                Disable All
              </button>
              <button
                onClick={() => setShowBranchPanel(false)}
                style={{
                  padding: '6px 12px',
                  background: '#475569',
                  border: 'none',
                  borderRadius: 6,
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: 12
                }}
              >
                Close
              </button>
            </div>
          </div>

          {/* Branch Grid */}
          <div style={{
            flex: 1,
            overflow: 'auto',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
            gap: 8,
            alignContent: 'start'
          }}>
            {filteredBranches.map(branch => {
              const isActive = activeBranches.has(branch.id);
              return (
                <button
                  key={branch.id}
                  onClick={() => handleToggleBranch(branch.id)}
                  title={`Branch ${branch.id}: Bus ${branch.from_bus} → ${branch.to_bus}`}
                  style={{
                    padding: '8px 6px',
                    background: isActive ? 'rgba(37, 99, 235, 0.3)' : '#1e293b',
                    border: `1px solid ${isActive ? '#3b82f6' : '#334155'}`,
                    borderRadius: 6,
                    color: isActive ? '#93c5fd' : '#64748b',
                    cursor: 'pointer',
                    fontSize: 12,
                    fontWeight: 500,
                    textDecoration: isActive ? 'none' : 'line-through'
                  }}
                >
                  {branch.from_bus}→{branch.to_bus}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
