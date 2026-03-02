import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0a0a0a",
        gold: {
          DEFAULT: "#c8a45c",
          dim: "#a68942",
          glow: "#d4b06a",
        },
        stone: {
          DEFAULT: "#3a3631",
          light: "#4a4540",
        },
        warm: {
          gray: "#8a8279",
          text: "#e8e2d8",
          muted: "#b0a899",
        },
      },
      fontFamily: {
        serif: ["Cinzel", "Georgia", "serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      animation: {
        "fade-in": "fadeIn 0.6s ease-out",
        "glow-pulse": "glowPulse 3s ease-in-out infinite",
        float: "float 6s ease-in-out infinite",
        shimmer: "shimmer 2.5s ease-in-out infinite",
        "pulse-ring": "pulseRing 2s ease-out infinite",
        typing: "cursorBlink 0.8s step-end infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        glowPulse: {
          "0%, 100%": { textShadow: "0 0 20px rgba(200, 164, 92, 0.3)" },
          "50%": { textShadow: "0 0 40px rgba(200, 164, 92, 0.6)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        pulseRing: {
          "0%": {
            boxShadow: "0 0 0 0 rgba(200, 164, 92, 0.4)",
          },
          "70%": {
            boxShadow: "0 0 0 12px rgba(200, 164, 92, 0)",
          },
          "100%": {
            boxShadow: "0 0 0 0 rgba(200, 164, 92, 0)",
          },
        },
        cursorBlink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
      },
      boxShadow: {
        "glow-gold": "0 0 15px rgba(200, 164, 92, 0.2), 0 0 5px rgba(200, 164, 92, 0.1)",
        "glow-gold-lg":
          "0 0 30px rgba(200, 164, 92, 0.3), 0 0 10px rgba(200, 164, 92, 0.15)",
      },
    },
  },
  plugins: [],
};
export default config;
