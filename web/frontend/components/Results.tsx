"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import ScoreBar from "./ui/ScoreBar";
import GoldDivider from "./ui/GoldDivider";
import ResultDetails from "./ResultDetails";
import { PipelineResult } from "@/lib/types";

interface ResultsProps {
  result: PipelineResult | null;
  isLoading: boolean;
}

function getMainOutput(result: PipelineResult): {
  dothraki: string;
  english: string;
  score: number;
} {
  const strategy = result.strategy;

  if (strategy === "phoneme" || strategy === "ensemble") {
    const dothraki =
      result.raw_dothraki ||
      result.translation?.words?.map((w) => w.dothraki || w.original).join(" ") ||
      "";
    const english = result.translation?.text || "";
    const words = result.translation?.words || [];
    const score =
      words.length > 0
        ? words.reduce((s, w) => s + w.confidence, 0) / words.length
        : 0;
    return { dothraki, english, score };
  }

  if (strategy === "embedding" || strategy === "dtw") {
    const top = result.clip_matches?.[0];
    return {
      dothraki: top?.dothraki || result.raw_dothraki || "",
      english: top?.english || "",
      score: top?.score || 0,
    };
  }

  if (strategy === "finetune") {
    return {
      dothraki: result.raw_dothraki || "",
      english: "",
      score: result.quality === "good" ? 0.7 : 0.3,
    };
  }

  return { dothraki: "", english: "", score: 0 };
}

/** Typewriter hook: reveals text one character at a time */
function useTypewriter(text: string, speed = 40, startDelay = 0) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);

  useEffect(() => {
    setDisplayed("");
    setDone(false);
    if (!text) { setDone(true); return; }

    let i = 0;
    let timeout: ReturnType<typeof setTimeout>;

    const startTimeout = setTimeout(() => {
      const tick = () => {
        if (i < text.length) {
          setDisplayed(text.slice(0, i + 1));
          i++;
          timeout = setTimeout(tick, speed);
        } else {
          setDone(true);
        }
      };
      tick();
    }, startDelay);

    return () => {
      clearTimeout(startTimeout);
      clearTimeout(timeout);
    };
  }, [text, speed, startDelay]);

  return { displayed, done };
}

export default function Results({ result, isLoading }: ResultsProps) {
  if (isLoading) {
    return (
      <div className="flex flex-col items-center py-12">
        <div className="w-8 h-8 border-2 border-gold/30 border-t-gold rounded-full animate-spin" />
        <p className="text-warm-muted mt-4 text-sm">Running pipeline&hellip;</p>
      </div>
    );
  }

  if (!result) return null;

  const { dothraki, english, score } = getMainOutput(result);

  return (
    <TypewriterResults
      dothraki={dothraki}
      english={english}
      score={score}
      result={result}
    />
  );
}

function TypewriterResults({
  dothraki,
  english,
  score,
  result,
}: {
  dothraki: string;
  english: string;
  score: number;
  result: PipelineResult;
}) {
  const { displayed: dothrakiText, done: dothrakiDone } = useTypewriter(dothraki, 35);
  const { displayed: englishText } = useTypewriter(english, 30, dothrakiDone ? 200 : 99999);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      <GoldDivider className="max-w-sm mx-auto" />

      <div className="text-center space-y-3">
        {dothraki && (
          <p className="font-serif text-2xl md:text-3xl text-gold leading-relaxed">
            {dothrakiText}
            {!dothrakiDone && (
              <span className="inline-block w-[2px] h-[1em] bg-gold ml-1 animate-typing align-middle" />
            )}
          </p>
        )}
        {english && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: dothrakiDone ? 1 : 0 }}
            transition={{ duration: 0.5 }}
            className="text-warm-muted text-lg italic"
          >
            {englishText}
          </motion.p>
        )}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: dothrakiDone ? 1 : 0, y: dothrakiDone ? 0 : 10 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="max-w-xs mx-auto"
      >
        <ScoreBar score={score} label="Confidence" />
      </motion.div>

      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{
          scale: dothrakiDone ? 1 : 0,
          opacity: dothrakiDone ? 1 : 0,
        }}
        transition={{ type: "spring", stiffness: 400, damping: 20, delay: 0.4 }}
        className="flex justify-center"
      >
        <span
          className={`text-xs uppercase tracking-wider px-3 py-1 rounded-full border ${
            result.quality === "good"
              ? "border-green-800/40 text-green-400/80 bg-green-900/20"
              : result.quality === "low_confidence"
              ? "border-yellow-800/40 text-yellow-400/80 bg-yellow-900/20"
              : "border-red-800/40 text-red-400/80 bg-red-900/20"
          }`}
        >
          {result.quality}
        </span>
      </motion.div>

      <ResultDetails result={result} />
    </motion.div>
  );
}
