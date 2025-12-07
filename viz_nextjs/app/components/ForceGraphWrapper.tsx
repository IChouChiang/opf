"use client";

import dynamic from 'next/dynamic';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-full bg-gray-100 text-gray-500">Loading Graph...</div>
});

export default ForceGraph2D;
