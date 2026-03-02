"use client";

import { motion } from "framer-motion";

interface ScoreBarProps {
  score: number; // 0-1
  label?: string;
  delay?: number;
}

export default function ScoreBar({ score, label, delay = 0 }: ScoreBarProps) {
  const pct = Math.round(score * 100);

  return (
    <div className="flex items-center gap-3">
      {label && <span className="text-sm text-warm-muted w-20 shrink-0">{label}</span>}
      <div className="flex-1 h-2 bg-stone rounded-full overflow-hidden relative">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: "easeOut", delay }}
          className="h-full bg-gradient-to-r from-gold-dim to-gold rounded-full relative"
          style={{
            backgroundSize: "200% 100%",
            animation: "shimmerSweep 3s ease-in-out infinite",
          }}
        >
          {/* Glow dot at the leading edge */}
          <div
            className="absolute right-0 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full"
            style={{
              background: "#d4b06a",
              boxShadow: "0 0 8px rgba(212,176,106,0.6), 0 0 3px rgba(212,176,106,0.4)",
            }}
          />
        </motion.div>
      </div>
      <span className="text-sm text-warm-muted w-12 text-right">{pct}%</span>
    </div>
  );
}
