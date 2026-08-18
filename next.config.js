/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async redirects() {
    // Single analysis surface: the Dashboard. Any /analyze request → /dashboard.
    return [
      { source: '/analyze', destination: '/dashboard', permanent: false },
    ]
  },
  webpack: (config, { isServer }) => {
    // Handle MediaPipe WASM files
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }
    return config;
  },
}

module.exports = nextConfig
