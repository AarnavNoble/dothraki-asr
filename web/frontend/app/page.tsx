"use client";

import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import Hero from "@/components/Hero";
import AudioPicker from "@/components/AudioPicker";
import { DemoClip } from "@/components/ClipCard";
import StrategySelector from "@/components/StrategySelector";
import Waveform from "@/components/Waveform";
import TranscribeButton from "@/components/TranscribeButton";
import Results from "@/components/Results";
import SpeakDothraki from "@/components/SpeakDothraki";
import About from "@/components/About";
import { PipelineResult } from "@/lib/types";

const sectionReveal = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: "easeOut" as const },
  },
};

export default function Home() {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [selectedClip, setSelectedClip] = useState<DemoClip | null>(null);
  const [strategy, setStrategy] = useState("phoneme");
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const uploadedFileRef = useRef<File | null>(null);

  const handleAudioSelect = useCallback(
    (url: string, clip: DemoClip | null) => {
      setAudioUrl(url);
      setSelectedClip(clip);
      setResult(null);
      setError(null);

      if (url.startsWith("blob:")) {
        fetch(url)
          .then((r) => r.blob())
          .then((blob) => {
            uploadedFileRef.current = new File([blob], "upload.wav", {
              type: blob.type,
            });
          });
      } else {
        uploadedFileRef.current = null;
      }
    },
    []
  );

  const handleTranscribe = async () => {
    if (!audioUrl) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      let res: Response;

      if (selectedClip) {
        const form = new FormData();
        form.append("clip_id", selectedClip.id);
        form.append("category", selectedClip.category);
        form.append("strategy", strategy);

        res = await fetch("/api/transcribe-clip", { method: "POST", body: form });
      } else if (uploadedFileRef.current) {
        const form = new FormData();
        form.append("audio", uploadedFileRef.current);
        form.append("strategy", strategy);

        res = await fetch("/api/transcribe", { method: "POST", body: form });
      } else {
        setError("No audio selected");
        setIsLoading(false);
        return;
      }

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Request failed" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      setResult(await res.json());
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Something went wrong";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen relative z-10">
      <Hero />

      <main className="max-w-4xl mx-auto px-4 pb-16 space-y-10">
        <motion.section
          variants={sectionReveal}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          className="glass-panel p-6"
        >
          <h2 className="font-serif text-xl text-gold mb-4 gold-accent-bar">
            Speak Dothraki
          </h2>
          <p className="text-warm-muted text-sm mb-4">
            Type English and hear it spoken in Dothraki with Khal Drogo&apos;s voice
          </p>
          <SpeakDothraki />
        </motion.section>

        <motion.section
          variants={sectionReveal}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          className="glass-panel p-6"
        >
          <h2 className="font-serif text-xl text-gold mb-4 gold-accent-bar">
            Choose Audio
          </h2>
          <AudioPicker
            onAudioSelect={handleAudioSelect}
            selectedClip={selectedClip}
          />
        </motion.section>

        {audioUrl && (
          <motion.section
            variants={sectionReveal}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-50px" }}
            className="glass-panel p-6"
          >
            <Waveform audioUrl={audioUrl} />
          </motion.section>
        )}

        <motion.section
          variants={sectionReveal}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          className="glass-panel p-6 flex flex-col sm:flex-row items-start sm:items-center gap-6"
        >
          <div className="flex-1">
            <h2 className="font-serif text-xl text-gold mb-3 gold-accent-bar">
              Strategy
            </h2>
            <StrategySelector selected={strategy} onSelect={setStrategy} />
          </div>
          <TranscribeButton
            onClick={handleTranscribe}
            isLoading={isLoading}
            disabled={!audioUrl}
          />
        </motion.section>

        {error && (
          <div className="border border-red-800/40 bg-red-900/20 rounded-lg p-4 text-red-400 text-sm">
            {error}
          </div>
        )}

        <motion.section
          variants={sectionReveal}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
        >
          <Results result={result} isLoading={isLoading} />
        </motion.section>
      </main>

      <About />
    </div>
  );
}
