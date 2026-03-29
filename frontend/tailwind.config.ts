import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: { extend: { fontFamily: { mono: ["DM Mono", "monospace"], sans: ["Sora", "sans-serif"] } } },
  plugins: [require("@tailwindcss/typography")],
};
export default config;
