"use client";

import { motion } from "framer-motion";

export interface DemoClip {
  id: string;
  audio_file: string;
  dothraki: string;
  english: string;
  category: string;
  ipa_clean?: string;
  scene?: string;
}

interface ClipCardProps {
  clip: DemoClip;
  isSelected: boolean;
  onSelect: (clip: DemoClip) => void;
}

export default function ClipCard({ clip, isSelected, onSelect }: ClipCardProps) {
  return (
    <motion.button
      whileHover={{ y: -2, transition: { duration: 0.2 } }}
      whileTap={{ scale: 0.98 }}
      onClick={() => onSelect(clip)}
      className={`relative text-left p-4 rounded-lg border transition-all duration-200 w-full overflow-hidden group ${
        isSelected
          ? "border-gold/60 bg-gold/10 shadow-glow-gold"
          : "border-stone hover:border-warm-gray bg-stone/30 hover:bg-stone/50"
      }`}
    >
      {/* Animated gradient border on hover */}
      <div
        className="absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{
          background:
            "conic-gradient(from 0deg, transparent, rgba(200,164,92,0.15), transparent, rgba(200,164,92,0.15), transparent)",
          animation: "spin 4s linear infinite",
          mask: "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
          maskComposite: "exclude",
          WebkitMaskComposite: "xor",
          padding: "1px",
        }}
      />

      <div className="flex items-start justify-between gap-2">
        <p className="font-serif text-sm text-gold truncate flex-1">
          {clip.dothraki || clip.english}
        </p>
        {/* Waveform icon that pulses on hover */}
        <span className="text-gold/40 group-hover:text-gold/70 transition-colors text-xs group-hover:animate-pulse">
          &#9835;
        </span>
      </div>
      <p className="text-xs text-warm-muted mt-1 truncate">{clip.english}</p>
      {clip.category === "raw" && (
        <span className="inline-block mt-2 text-[10px] uppercase tracking-wider text-warm-gray bg-stone px-2 py-0.5 rounded">
          Real Audio
        </span>
      )}
    </motion.button>
  );
}
