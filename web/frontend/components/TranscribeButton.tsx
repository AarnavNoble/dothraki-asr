"use client";

import { motion } from "framer-motion";

interface TranscribeButtonProps {
  onClick: () => void;
  isLoading: boolean;
  disabled: boolean;
}

function EqualizerBars() {
  const delays = [0, 0.15, 0.3, 0.1, 0.25];
  return (
    <span className="flex items-end gap-[3px] h-5">
      {delays.map((d, i) => (
        <span
          key={i}
          className="w-[3px] bg-bg rounded-full origin-bottom"
          style={{
            height: "100%",
            animation: `eqBounce ${0.5 + i * 0.1}s ease-in-out ${d}s infinite`,
          }}
        />
      ))}
    </span>
  );
}

export default function TranscribeButton({
  onClick,
  isLoading,
  disabled,
}: TranscribeButtonProps) {
  return (
    <div className="relative">
      {/* Pulsing ring when idle + enabled */}
      {!disabled && !isLoading && (
        <div className="absolute inset-0 rounded-lg animate-pulse-ring pointer-events-none" />
      )}

      <motion.button
        whileHover={!disabled ? { scale: 1.03 } : {}}
        whileTap={!disabled ? { scale: 0.97 } : {}}
        onClick={onClick}
        disabled={disabled || isLoading}
        className={`relative px-8 py-3 rounded-lg font-serif text-lg tracking-wide transition-all duration-300 overflow-hidden ${
          disabled
            ? "bg-stone text-warm-gray cursor-not-allowed opacity-60"
            : "bg-gradient-to-r from-gold-dim to-gold text-bg hover:shadow-glow-gold-lg"
        }`}
      >
        {/* Shimmer sweep on hover */}
        {!disabled && !isLoading && (
          <div
            className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity"
            style={{
              background:
                "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.15) 50%, transparent 100%)",
              backgroundSize: "200% 100%",
              animation: "shimmerSweep 2s ease-in-out infinite",
            }}
          />
        )}

        {isLoading ? (
          <span className="flex items-center gap-3 relative z-10">
            <EqualizerBars />
            Transcribing&hellip;
          </span>
        ) : (
          <span className="relative z-10">Transcribe</span>
        )}
      </motion.button>
    </div>
  );
}
