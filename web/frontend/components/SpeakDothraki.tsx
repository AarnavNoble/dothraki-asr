"use client";

import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface SpeakResult {
  dothraki: string;
  english: string;
  score: number;
  audio_url: string;
  alternatives: { dothraki: string; english: string; score: number }[];
}

function EqualizerBars() {
  const delays = [0, 0.15, 0.3, 0.1, 0.25];
  return (
    <span className="flex items-end gap-[3px] h-4">
      {delays.map((d, i) => (
        <span
          key={i}
          className="w-[3px] bg-gold rounded-full origin-bottom"
          style={{
            height: "100%",
            animation: `eqBounce ${0.5 + i * 0.1}s ease-in-out ${d}s infinite`,
          }}
        />
      ))}
    </span>
  );
}

export default function SpeakDothraki() {
  const [text, setText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SpeakResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handleSpeak = useCallback(async () => {
    if (!text.trim() || isLoading) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("text", text.trim());

      const res = await fetch("/api/speak", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Request failed" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data: SpeakResult = await res.json();
      setResult(data);

      // Auto-play
      if (audioRef.current) {
        audioRef.current.pause();
      }
      const audio = new Audio(data.audio_url);
      audioRef.current = audio;
      audio.onplay = () => setIsPlaying(true);
      audio.onended = () => setIsPlaying(false);
      audio.onpause = () => setIsPlaying(false);
      audio.play().catch(() => {});
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Something went wrong";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [text, isLoading]);

  const handleReplay = () => {
    if (result && result.audio_url) {
      if (audioRef.current) {
        audioRef.current.currentTime = 0;
        audioRef.current.play().catch(() => {});
      }
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-3">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSpeak()}
          placeholder="Type something in English..."
          className="flex-1 bg-stone/30 border border-stone rounded-lg px-4 py-3 text-warm-text placeholder:text-warm-gray focus:outline-none focus:border-gold/40 transition-colors"
          disabled={isLoading}
          maxLength={500}
        />
        <motion.button
          whileHover={!isLoading ? { scale: 1.03 } : {}}
          whileTap={!isLoading ? { scale: 0.97 } : {}}
          onClick={handleSpeak}
          disabled={!text.trim() || isLoading}
          className={`px-6 py-3 rounded-lg font-serif tracking-wide transition-all duration-300 flex items-center gap-2 ${
            !text.trim() || isLoading
              ? "bg-stone text-warm-gray cursor-not-allowed opacity-60"
              : "bg-gradient-to-r from-gold-dim to-gold text-bg hover:shadow-glow-gold-lg"
          }`}
        >
          {isLoading ? (
            <>
              <EqualizerBars />
              <span>Generating&hellip;</span>
            </>
          ) : (
            <>
              <span className="text-lg">&#9654;</span>
              <span>Speak</span>
            </>
          )}
        </motion.button>
      </div>

      {error && (
        <div className="border border-red-800/40 bg-red-900/20 rounded-lg p-3 text-red-400 text-sm">
          {error}
        </div>
      )}

      <AnimatePresence mode="wait">
        {isLoading && (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-center py-6"
          >
            <p className="text-warm-muted text-sm">
              Finding translation and generating speech&hellip;
            </p>
            <p className="text-warm-gray text-xs mt-1">
              This takes ~15 seconds on first use
            </p>
          </motion.div>
        )}

        {result && !isLoading && (
          <motion.div
            key="result"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            className="border border-gold/20 rounded-lg p-5 space-y-4 bg-gold/[0.03]"
          >
            {/* Dothraki output */}
            <div className="text-center space-y-2">
              <p className="font-serif text-2xl text-gold leading-relaxed">
                {result.dothraki}
              </p>
              <p className="text-warm-muted text-sm italic">
                &ldquo;{result.english}&rdquo;
              </p>
            </div>

            {/* Playback controls */}
            <div className="flex items-center justify-center gap-3">
              <button
                onClick={handleReplay}
                className="relative w-10 h-10 rounded-full border border-gold/40 flex items-center justify-center text-gold hover:bg-gold/10 transition-colors"
              >
                {isPlaying && (
                  <motion.div
                    className="absolute inset-0 rounded-full"
                    animate={{
                      boxShadow: [
                        "0 0 0 0 rgba(200,164,92,0.4)",
                        "0 0 0 8px rgba(200,164,92,0)",
                      ],
                    }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                )}
                <span className="relative z-10">
                  {isPlaying ? "\u275A\u275A" : "\u25B6"}
                </span>
              </button>
              <span className="text-xs text-warm-gray">
                Match: {Math.round(result.score * 100)}%
              </span>
            </div>

            {/* Alternatives */}
            {result.alternatives.length > 0 && (
              <div className="pt-2 border-t border-stone/50">
                <p className="text-xs text-warm-gray uppercase tracking-wider mb-2">
                  Other matches
                </p>
                <div className="space-y-1">
                  {result.alternatives.map((alt, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 text-xs text-warm-muted"
                    >
                      <span className="text-gold/60 font-serif truncate flex-1">
                        {alt.dothraki}
                      </span>
                      <span className="text-warm-gray shrink-0">
                        {Math.round(alt.score * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
