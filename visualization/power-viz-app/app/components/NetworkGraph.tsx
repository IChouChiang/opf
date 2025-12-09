// Network Graph Component - Full screen D3.js Force Simulation

'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { NetworkNode, NetworkLink } from '../types/power-system';

interface NetworkGraphProps {
    nodes: NetworkNode[];
    links: NetworkLink[];
}

export default function NetworkGraph({ nodes, links }: NetworkGraphProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const simulationRef = useRef<d3.Simulation<NetworkNode, NetworkLink> | null>(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

    // Track container size
    useEffect(() => {
        if (!containerRef.current) return;

        const updateDimensions = () => {
            if (containerRef.current) {
                const { width, height } = containerRef.current.getBoundingClientRect();
                if (width > 0 && height > 0) {
                    setDimensions({ width, height });
                }
            }
        };

        updateDimensions();

        const resizeObserver = new ResizeObserver(() => updateDimensions());
        resizeObserver.observe(containerRef.current);

        return () => resizeObserver.disconnect();
    }, []);

    useEffect(() => {
        if (!svgRef.current || nodes.length === 0) return;

        // Clear previous content
        d3.select(svgRef.current).selectAll('*').remove();

        const svg = d3.select(svgRef.current);
        const { width, height } = dimensions;

        // Create zoom behavior
        const g = svg.append('g');

        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.2, 3])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });

        svg.call(zoom);

        // Initial zoom to center
        svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(0.9));

        // Color schemes
        const busTypeColors: Record<string, string> = {
            PQ: '#60a5fa',
            PV: '#34d399',
            Slack: '#f87171',
            Isolated: '#6b7280',
            Unknown: '#6b7280',
        };

        const islandColors = d3.schemeCategory10;

        // Filter active links
        const activeLinks = links.filter(link => link.active !== false);

        // Create force simulation
        const simulation = d3.forceSimulation<NetworkNode>(nodes)
            .force('link', d3.forceLink<NetworkNode, NetworkLink>(activeLinks)
                .id(d => d.id)
                .distance(100))
            .force('charge', d3.forceManyBody<NetworkNode>().strength(-400))
            .force('center', d3.forceCenter(0, 0))
            .force('collision', d3.forceCollide<NetworkNode>().radius(30));

        simulationRef.current = simulation;

        // Draw links
        const link = g.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(activeLinks)
            .enter()
            .append('line')
            .attr('stroke', d => {
                const sourceNode = typeof d.source === 'number' ? nodes.find(n => n.id === d.source) : d.source;
                if (sourceNode?.island !== undefined && sourceNode.island > 0) {
                    return islandColors[sourceNode.island % islandColors.length];
                }
                return '#475569';
            })
            .attr('stroke-width', 2.5)
            .attr('stroke-opacity', 0.7);

        // Draw nodes
        const node = g.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(nodes)
            .enter()
            .append('g')
            .attr('cursor', 'grab')
            .call(d3.drag<SVGGElement, NetworkNode>()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended) as any);

        // Node shapes based on type
        node.each(function (d) {
            const el = d3.select(this);
            const color = d.island !== undefined && d.island > 0
                ? islandColors[d.island % islandColors.length]
                : busTypeColors[d.type_name] || busTypeColors.Unknown;

            if (d.type_name === 'Slack') {
                // Star shape
                const points = 5;
                const outerRadius = 14;
                const innerRadius = 7;
                let pathData = '';
                for (let i = 0; i < points * 2; i++) {
                    const radius = i % 2 === 0 ? outerRadius : innerRadius;
                    const angle = (i * Math.PI / points) - Math.PI / 2;
                    const x = radius * Math.cos(angle);
                    const y = radius * Math.sin(angle);
                    pathData += `${i === 0 ? 'M' : 'L'} ${x},${y} `;
                }
                pathData += 'Z';

                el.append('path')
                    .attr('d', pathData)
                    .attr('fill', color)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2);
            } else if (d.type_name === 'PV') {
                // Diamond shape
                el.append('rect')
                    .attr('x', -10)
                    .attr('y', -10)
                    .attr('width', 20)
                    .attr('height', 20)
                    .attr('transform', 'rotate(45)')
                    .attr('fill', color)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2);
            } else {
                // Circle for PQ
                el.append('circle')
                    .attr('r', 12)
                    .attr('fill', color)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2);
            }

            // Generator indicator
            if (d.hasGenerator) {
                el.append('circle')
                    .attr('r', 5)
                    .attr('fill', '#fbbf24')
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 1);
            }

            // Bus number label
            el.append('text')
                .text(d.id)
                .attr('y', 28)
                .attr('text-anchor', 'middle')
                .attr('font-size', '11px')
                .attr('font-weight', '600')
                .attr('fill', '#e2e8f0')
                .attr('class', 'pointer-events-none select-none');
        });

        // Tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'fixed hidden bg-slate-900/95 text-white px-3 py-2 rounded-lg text-sm border border-slate-600 shadow-xl z-50 pointer-events-none')
            .style('backdrop-filter', 'blur(8px)');

        node.on('mouseenter', function (event, d) {
            d3.select(this).select('circle, rect, path').attr('stroke-width', 3);

            tooltip.html(`
                <div class="font-semibold text-blue-400 mb-1">Bus ${d.id}</div>
                <div class="text-xs space-y-0.5">
                    <div>Type: <span class="text-white">${d.type_name}</span></div>
                    <div>Load: <span class="text-white">${d.pd.toFixed(1)} MW</span></div>
                    <div>Voltage: <span class="text-white">${d.vm.toFixed(3)} pu</span></div>
                    ${d.island !== undefined && d.island > 0 ? `<div class="text-amber-400">Island ${d.island + 1}</div>` : ''}
                </div>
            `)
                .style('left', (event.pageX + 12) + 'px')
                .style('top', (event.pageY - 12) + 'px')
                .classed('hidden', false);
        })
            .on('mousemove', function (event) {
                tooltip
                    .style('left', (event.pageX + 12) + 'px')
                    .style('top', (event.pageY - 12) + 'px');
            })
            .on('mouseleave', function () {
                d3.select(this).select('circle, rect, path').attr('stroke-width', 2);
                tooltip.classed('hidden', true);
            });

        // Update positions on tick
        simulation.on('tick', () => {
            link
                .attr('x1', d => (typeof d.source === 'number' ? 0 : d.source.x) || 0)
                .attr('y1', d => (typeof d.source === 'number' ? 0 : d.source.y) || 0)
                .attr('x2', d => (typeof d.target === 'number' ? 0 : d.target.x) || 0)
                .attr('y2', d => (typeof d.target === 'number' ? 0 : d.target.y) || 0);

            node.attr('transform', d => `translate(${d.x || 0},${d.y || 0})`);
        });

        function dragstarted(event: any, d: NetworkNode) {
            if (!event.active && simulationRef.current) simulationRef.current.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event: any, d: NetworkNode) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event: any, d: NetworkNode) {
            if (!event.active && simulationRef.current) simulationRef.current.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        return () => {
            tooltip.remove();
            if (simulationRef.current) {
                simulationRef.current.stop();
            }
        };
    }, [nodes, links, dimensions]);

    return (
        <div ref={containerRef} className="w-full h-full">
            <svg
                ref={svgRef}
                width={dimensions.width}
                height={dimensions.height}
                className="w-full h-full"
            />
        </div>
    );
}
