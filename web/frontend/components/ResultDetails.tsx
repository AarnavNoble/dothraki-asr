"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ScoreBar from "./ui/ScoreBar";
import { PipelineResult } from "@/lib/types";

interface ResultDetailsProps {
  result: PipelineResult;
}

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, x: -10 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.3 } },
};

export default function ResultDetails({ result }: ResultDetailsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const strategy = result.strategy;

  return (
    <div className="border border-stone rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-5 py-3 flex items-center justify-between text-sm text-warm-muted hover:bg-stone/30 transition-colors"
      >
        <span>Strategy Details</span>
        <motion.span
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          &#x25BE;
        </motion.span>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="overflow-hidden"
          >
            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              className="px-5 pb-4 space-y-3"
            >
              {/* Phoneme strategy: per-word matches */}
              {strategy === "phoneme" && result.translation?.words && (
                <motion.div variants={itemVariants}>
                  <p className="text-xs text-warm-gray uppercase tracking-wider mb-2">
                    Per-word Matches
                  </p>
                  <div className="space-y-2">
                    {result.translation.words.map((w, i) => (
                      <motion.div
                        key={i}
                        variants={itemVariants}
                        className="flex items-center gap-3 text-sm"
                      >
                        <span className="text-warm-muted w-28 truncate font-mono text-xs">
                          {w.original}
                        </span>
                        <span className="text-gold">&rarr;</span>
                        <span className="text-warm-text">
                          {w.dothraki || "\u2014"}
                        </span>
                        <span className="text-warm-gray text-xs">
                          {w.english || ""}
                        </span>
                        <div className="flex-1">
                          <ScoreBar score={w.confidence} delay={i * 0.08} />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Whisper transcription */}
              {result.transcription && (
                <motion.div variants={itemVariants}>
                  <p className="text-xs text-warm-gray uppercase tracking-wider mb-1">
                    Whisper Transcription
                  </p>
                  <p className="text-sm text-warm-muted font-mono bg-stone/30 p-2 rounded">
                    {result.transcription.text}
                  </p>
                  <p className="text-xs text-warm-gray mt-1">
                    Model: {result.transcription.model} &middot; Language:{" "}
                    {result.transcription.language || "auto"}
                  </p>
                </motion.div>
              )}

              {/* Clip matches (embedding/dtw/ensemble) */}
              {result.clip_matches && result.clip_matches.length > 0 && (
                <motion.div variants={itemVariants}>
                  <p className="text-xs text-warm-gray uppercase tracking-wider mb-2">
                    Top Clip Matches
                  </p>
                  <div className="space-y-2">
                    {result.clip_matches.slice(0, 5).map((m, i) => (
                      <motion.div
                        key={i}
                        variants={itemVariants}
                        className="flex items-center gap-3 bg-stone/20 rounded p-2"
                      >
                        <span className="text-xs text-warm-gray w-12">
                          {m.clip_id}
                        </span>
                        <span className="text-sm text-gold flex-1 truncate">
                          {m.dothraki}
                        </span>
                        <div className="w-32">
                          <ScoreBar score={m.score} delay={i * 0.1} />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Finetune raw output */}
              {strategy === "finetune" && result.raw_dothraki && (
                <motion.div variants={itemVariants}>
                  <p className="text-xs text-warm-gray uppercase tracking-wider mb-1">
                    Raw Decoder Output
                  </p>
                  <p className="text-sm text-warm-muted font-mono bg-stone/30 p-2 rounded">
                    {result.raw_dothraki}
                  </p>
                </motion.div>
              )}

              {/* Ensemble note */}
              {strategy === "ensemble" && (
                <motion.div variants={itemVariants}>
                  <p className="text-xs text-warm-gray uppercase tracking-wider mb-1">
                    Ensemble: combined phoneme, embedding, DTW, and fine-tuned strategies
                  </p>
                </motion.div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
