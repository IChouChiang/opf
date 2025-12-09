// Graph analysis utilities for network connectivity

import { NetworkNode, NetworkLink, ConnectivityResult } from '../types/power-system';

/**
 * Detect connected components in the network using BFS
 * @param nodes Array of network nodes
 * @param links Array of active network links
 * @returns Connectivity analysis result
 */
export function analyzeConnectivity(
    nodes: NetworkNode[],
    links: NetworkLink[]
): ConnectivityResult {
    const n = nodes.length;
    const visited = new Array(n).fill(false);
    const islands: number[][] = [];

    // Build adjacency list from active links
    const adj: Map<number, Set<number>> = new Map();

    nodes.forEach((node) => {
        adj.set(node.id, new Set());
    });

    links.forEach((link) => {
        if (link.active !== false) {
            const sourceId = typeof link.source === 'number' ? link.source : link.source.id;
            const targetId = typeof link.target === 'number' ? link.target : link.target.id;

            adj.get(sourceId)?.add(targetId);
            adj.get(targetId)?.add(sourceId);
        }
    });

    // BFS to find connected components
    const bfs = (startBusId: number): number[] => {
        const component: number[] = [];
        const queue: number[] = [startBusId];
        const startIndex = nodes.findIndex(n => n.id === startBusId);

        if (startIndex >= 0) {
            visited[startIndex] = true;
        }

        while (queue.length > 0) {
            const busId = queue.shift()!;
            component.push(busId);

            const neighbors = adj.get(busId) || new Set();
            neighbors.forEach((neighborId) => {
                const neighborIndex = nodes.findIndex(n => n.id === neighborId);
                if (neighborIndex >= 0 && !visited[neighborIndex]) {
                    visited[neighborIndex] = true;
                    queue.push(neighborId);
                }
            });
        }

        return component;
    };

    // Find all connected components
    nodes.forEach((node, idx) => {
        if (!visited[idx]) {
            const component = bfs(node.id);
            if (component.length > 0) {
                islands.push(component);
            }
        }
    });

    return {
        isConnected: islands.length <= 1,
        numIslands: islands.length,
        islands,
    };
}

/**
 * Assign island numbers to nodes
 */
export function assignIslands(
    nodes: NetworkNode[],
    islands: number[][]
): void {
    nodes.forEach(node => {
        node.island = undefined;
    });

    islands.forEach((island, islandIdx) => {
        island.forEach(busId => {
            const node = nodes.find(n => n.id === busId);
            if (node) {
                node.island = islandIdx;
            }
        });
    });
}
