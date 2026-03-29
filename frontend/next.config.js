/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**.amazonaws.com",
      },
    ],
  },
  async headers() {
    const isDev = process.env.NODE_ENV !== "production";
    // framer-motion and three.js/@react-three use new Function() / eval internally.
    // 'unsafe-eval' is required in dev to silence the CSP violation.
    // In production builds these are pre-compiled and do not need eval.
    const scriptSrc = isDev
      ? "script-src 'self' 'unsafe-inline' 'unsafe-eval'"
      : "script-src 'self' 'unsafe-inline'";
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              scriptSrc,
              "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
              "font-src 'self' https://fonts.gstatic.com",
              "img-src 'self' data: blob: https://*.amazonaws.com",
              "connect-src 'self' http://localhost:8000 https://arken.in wss://arken.in",
              "worker-src blob:",
            ].join("; "),
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;