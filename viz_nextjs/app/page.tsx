"use client";

import { useState, useEffect, useMemo, useCallback } from 'react';
import ForceGraph2D from './components/ForceGraphWrapper';
import { Settings, RefreshCw, AlertTriangle, CheckCircle, Scissors } from 'lucide-react';

// Types
interface Node {
  id: number;
  label: string;
  type: 'Slack' | 'PV' | 'PQ';
  pd: number;
  qd: number;
  pg: number;
  qg: number;
  val?: number; // for visual size
  color?: string;
}

interface Edge {
  id: number;
  source: number | Node; // ForceGraph converts numbers to objects
  target: number | Node;
  r: number;
  x: number;
  b: number;
  rateA: number;
}

interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

export default function Home() {
  const [selectedCase, setSelectedCase] = useState<string>('case14');
  const [rawData, setRawData] = useState<GraphData | null>(null);
  const [cutBranches, setCutBranches] = useState<Set<number>>(new Set());
  const [targetCutCount, setTargetCutCount] = useState<number>(1);
  const [islands, setIslands] = useState<number>(1);
  const [showCutLines, setShowCutLines] = useState<boolean>(true);

  // Load Data
  useEffect(() => {
    setRawData(null); // Clear previous data while loading
    setCutBranches(new Set()); // Reset cuts when case changes
    
    fetch(`/data/${selectedCase}.json`)
      .then(res => res.json())
      .then(data => {
        // Process nodes for visualization
        const nodes = data.nodes.map((n: any) => ({
          ...n,
          val: n.type === 'Slack' ? 15 : n.type === 'PV' ? 10 : 5,
          color: n.type === 'Slack' ? '#ef4444' : n.type === 'PV' ? '#22c55e' : '#3b82f6'
        }));
        setRawData({ nodes, edges: data.edges });
      })
      .catch(err => console.error("Failed to load case data:", err));
  }, [selectedCase]);

  // Toggle Branch Cut
  const handleLinkClick = useCallback((link: any) => {
    const id = link.id;
    setCutBranches(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  // Reset
  const handleReset = () => setCutBranches(new Set());

  // Compute Active Graph & Connectivity
  const { activeEdges, activeNodes, connectivityStatus } = useMemo(() => {
    if (!rawData) return { activeEdges: [], activeNodes: [], connectivityStatus: { islands: 0, isSplit: false } };

    // Filter edges
    // If showCutLines is true, we keep them but style them differently (handled in render)
    // But for connectivity, we must ignore cut branches.
    
    const activeEdgesList = rawData.edges.filter(e => !cutBranches.has(e.id));
    
    // Connectivity Check (BFS)
    const adj = new Map<number, number[]>();
    rawData.nodes.forEach(n => adj.set(n.id, []));
    
    activeEdgesList.forEach(e => {
      // Handle both object ref (after graph init) and number id (initial)
      const s = typeof e.source === 'object' ? (e.source as Node).id : e.source as number;
      const t = typeof e.target === 'object' ? (e.target as Node).id : e.target as number;
      adj.get(s)?.push(t);
      adj.get(t)?.push(s);
    });

    let visited = new Set<number>();
    let islandCount = 0;

    rawData.nodes.forEach(node => {
      if (!visited.has(node.id)) {
        islandCount++;
        // BFS
        const queue = [node.id];
        visited.add(node.id);
        while (queue.length > 0) {
          const u = queue.shift()!;
          const neighbors = adj.get(u) || [];
          for (const v of neighbors) {
            if (!visited.has(v)) {
              visited.add(v);
              queue.push(v);
            }
          }
        }
      }
    });

    return {
      activeEdges: showCutLines ? rawData.edges : activeEdgesList,
      activeNodes: rawData.nodes,
      connectivityStatus: { islands: islandCount, isSplit: islandCount > 1 }
    };
  }, [rawData, cutBranches, showCutLines]);

  useEffect(() => {
    setIslands(connectivityStatus.islands);
  }, [connectivityStatus]);

  if (!rawData) return <div className="flex h-screen items-center justify-center">Loading Case Data...</div>;

  return (
    <div className="flex h-screen w-full bg-gray-50 text-gray-900 font-sans overflow-hidden">
      
      {/* Sidebar */}
      <aside className="w-80 bg-white border-r border-gray-200 flex flex-col shadow-sm z-10">
        <div className="p-6 border-b border-gray-100">
          <h1 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <Settings className="w-5 h-5" />
            OPF Topology Viz
          </h1>
          <p className="text-xs text-gray-500 mt-1">N-k Contingency Analysis</p>
        </div>

        <div className="p-6 flex-1 overflow-y-auto space-y-8">
          
          {/* Case Selection */}
          <section>
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Select Case</h2>
            <select 
              value={selectedCase}
              onChange={(e) => setSelectedCase(e.target.value)}
              className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
            >
              <option value="case6ww">Case 6ww (Wood & Wollenberg)</option>
              <option value="case9">Case 9 (IEEE 9-Bus)</option>
              <option value="case14">Case 14 (IEEE 14-Bus)</option>
            </select>
          </section>

          {/* Settings Section */}
          <section>
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Settings</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Target Cutoff Count</label>
                <div className="flex items-center gap-2">
                  <input 
                    type="number" 
                    min="0"
                    value={targetCutCount}
                    onChange={(e) => setTargetCutCount(parseInt(e.target.value) || 0)}
                    className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
                  />
                </div>
              </div>

              <div className="flex items-center gap-2">
                <input 
                  type="checkbox" 
                  id="showCuts"
                  checked={showCutLines}
                  onChange={e => setShowCutLines(e.target.checked)}
                  className="rounded text-blue-600 focus:ring-blue-500"
                />
                <label htmlFor="showCuts" className="text-sm text-gray-700">Show Cut Lines (Dashed)</label>
              </div>
            </div>
          </section>

          {/* Status Section */}
          <section>
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">System Status</h2>
            
            <div className={`p-4 rounded-lg border ${connectivityStatus.isSplit ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
              <div className="flex items-center gap-2 mb-2">
                {connectivityStatus.isSplit ? (
                  <AlertTriangle className="w-5 h-5 text-red-600" />
                ) : (
                  <CheckCircle className="w-5 h-5 text-green-600" />
                )}
                <span className={`font-medium ${connectivityStatus.isSplit ? 'text-red-800' : 'text-green-800'}`}>
                  {connectivityStatus.isSplit ? 'System Split!' : 'System Connected'}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                Islands Detected: <span className="font-bold">{islands}</span>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="bg-gray-50 p-3 rounded border border-gray-200">
                <div className="text-xs text-gray-500">Current Cuts</div>
                <div className={`text-xl font-bold ${cutBranches.size === targetCutCount ? 'text-green-600' : 'text-gray-800'}`}>
                  {cutBranches.size} <span className="text-gray-400 text-sm">/ {targetCutCount}</span>
                </div>
              </div>
            </div>
          </section>

          {/* Cut List Section */}
          <section>
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Cut Branches</h2>
            {cutBranches.size === 0 ? (
              <div className="text-sm text-gray-400 italic">No branches cut. Click on a line to cut it.</div>
            ) : (
              <ul className="space-y-2">
                {Array.from(cutBranches).map(id => {
                  const edge = rawData.edges.find(e => e.id === id);
                  if (!edge) return null;
                  // Handle source/target being objects or numbers
                  const s = typeof edge.source === 'object' ? (edge.source as Node).id : edge.source;
                  const t = typeof edge.target === 'object' ? (edge.target as Node).id : edge.target;
                  return (
                    <li key={id} className="flex items-center justify-between bg-white p-2 rounded border border-gray-200 shadow-sm text-sm">
                      <span className="flex items-center gap-2">
                        <Scissors className="w-3 h-3 text-red-500" />
                        <span>Bus {s} ↔ Bus {t}</span>
                      </span>
                      <button 
                        onClick={() => handleLinkClick({id})}
                        className="text-gray-400 hover:text-red-600"
                      >
                        ×
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

        </div>

        <div className="p-6 border-t border-gray-200">
          <button 
            onClick={handleReset}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-gray-800 hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
          >
            <RefreshCw className="w-4 h-4" />
            Reset Topology
          </button>
        </div>
      </aside>

      {/* Main Graph Area */}
      <main className="flex-1 relative bg-gray-50">
        <ForceGraph2D
          graphData={{ nodes: activeNodes, edges: activeEdges }}
          nodeLabel="label"
          nodeColor="color"
          nodeRelSize={6}
          
          // Link Styling
          linkColor={useCallback((link: any) => cutBranches.has(link.id) ? '#ff0000' : '#999', [cutBranches])}
          linkWidth={useCallback((link: any) => cutBranches.has(link.id) ? 2 : 1, [cutBranches])}
          linkLineDash={useCallback((link: any) => cutBranches.has(link.id) ? [5, 5] : null, [cutBranches])}
          linkDirectionalParticles={useCallback((link: any) => cutBranches.has(link.id) ? 0 : 2, [cutBranches])}
          linkDirectionalParticleSpeed={0.005}
          
          // Interaction
          onLinkClick={handleLinkClick}
          
          // Canvas Config
          backgroundColor="#f9fafb"
        />
        
        {/* Overlay Legend */}
        <div className="absolute top-4 right-4 bg-white/90 backdrop-blur p-4 rounded-lg shadow-sm border border-gray-200 text-xs space-y-2">
          <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-red-500"></span> Slack Bus</div>
          <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-green-500"></span> PV Bus (Gen)</div>
          <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-blue-500"></span> PQ Bus (Load)</div>
          <div className="h-px bg-gray-200 my-2"></div>
          <div className="flex items-center gap-2"><span className="w-8 h-0.5 bg-gray-400"></span> Active Line</div>
          <div className="flex items-center gap-2"><span className="w-8 h-0.5 border-t-2 border-dashed border-red-500"></span> Cut Line</div>
        </div>
      </main>
    </div>
  );
}
