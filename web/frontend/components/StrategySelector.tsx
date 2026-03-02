"use client";

import { motion } from "framer-motion";

const STRATEGIES = [
  { id: "phoneme", label: "Phoneme", icon: "\u223F", desc: "IPA phoneme matching" },
  { id: "embedding", label: "Embedding", icon: "\u2B22", desc: "Semantic similarity" },
  { id: "dtw", label: "DTW", icon: "\u2248", desc: "Time-warped alignment" },
  { id: "finetune", label: "Fine-tuned", icon: "\u2699", desc: "Trained decoder" },
  { id: "ensemble", label: "Ensemble", icon: "\u2726", desc: "All strategies combined" },
] as const;

interface StrategySelectorProps {
  selected: string;
  onSelect: (strategy: string) => void;
}

export default function StrategySelector({ selected, onSelect }: StrategySelectorProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {STRATEGIES.map((s) => (
        <button
          key={s.id}
          onClick={() => onSelect(s.id)}
          className={`relative px-4 py-2 rounded-full text-sm transition-colors duration-200 border ${
            selected === s.id
              ? "text-gold border-transparent"
              : "text-warm-muted border-stone hover:border-warm-gray hover:text-warm-text"
          }`}
          title={s.desc}
        >
          {selected === s.id && (
            <motion.div
              layoutId="strategy-pill"
              className="absolute inset-0 bg-gold/20 border border-gold/40 rounded-full shadow-glow-gold"
              transition={{ type: "spring", stiffness: 400, damping: 30 }}
            />
          )}
          <span className="relative z-10 flex items-center gap-1.5">
            <span className="text-xs">{s.icon}</span>
            {s.label}
          </span>
        </button>
      ))}
    </div>
  );
}
