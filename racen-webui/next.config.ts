import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/api/answer", destination: "http://127.0.0.1:8011/answer" },
      { source: "/api/health", destination: "http://127.0.0.1:8011/health" },
    ];
  },
};

export default nextConfig;
