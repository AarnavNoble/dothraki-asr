"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import type WaveSurferType from "wavesurfer.js";

interface WaveformProps {
  audioUrl: string | null;
}

export default function Waveform({ audioUrl }: WaveformProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurferType | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!audioUrl || !containerRef.current) return;

    let ws: WaveSurferType | null = null;

    const init = async () => {
      const WaveSurfer = (await import("wavesurfer.js")).default;

      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
      }

      ws = WaveSurfer.create({
        container: containerRef.current!,
        waveColor: "#3a3631",
        progressColor: "#c8a45c",
        cursorColor: "#c8a45c",
        cursorWidth: 1,
        barWidth: 2,
        barGap: 2,
        barRadius: 2,
        height: 80,
        normalize: true,
      });

      ws.on("ready", () => setDuration(ws!.getDuration()));
      ws.on("audioprocess", () => setCurrentTime(ws!.getCurrentTime()));
      ws.on("seeking", () => setCurrentTime(ws!.getCurrentTime()));
      ws.on("play", () => setIsPlaying(true));
      ws.on("pause", () => setIsPlaying(false));
      ws.on("finish", () => setIsPlaying(false));

      ws.load(audioUrl);
      wavesurferRef.current = ws;
    };

    init();

    return () => {
      if (ws) ws.destroy();
    };
  }, [audioUrl]);

  const formatTime = (t: number) => {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  if (!audioUrl) return null;

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="w-full space-y-3">
      <div className="border border-gold/15 rounded-lg bg-stone/20 shadow-glow-gold overflow-hidden">
        <div
          ref={containerRef}
          className="waveform-container p-3"
        />
        {/* Seek bar / timeline */}
        <div className="h-1 bg-stone/50 mx-3 mb-3 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-gold-dim to-gold rounded-full transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button
          onClick={() => wavesurferRef.current?.playPause()}
          className="relative w-12 h-12 rounded-full border border-gold/40 flex items-center justify-center text-gold hover:bg-gold/10 transition-colors"
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
          <span className="relative z-10 text-lg">
            {isPlaying ? "\u275A\u275A" : "\u25B6"}
          </span>
        </button>
        <span className="text-sm text-warm-muted font-mono">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  );
}
