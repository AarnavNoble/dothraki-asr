"use client";

import { useCallback, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ClipCard, { DemoClip } from "./ClipCard";

type Tab = "demos" | "upload";

interface AudioPickerProps {
  onAudioSelect: (audioUrl: string, clip: DemoClip | null) => void;
  selectedClip: DemoClip | null;
}

const cardVariants = {
  hidden: { opacity: 0, y: 15 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.06, duration: 0.35, ease: "easeOut" as const },
  }),
};

export default function AudioPicker({ onAudioSelect, selectedClip }: AudioPickerProps) {
  const [tab, setTab] = useState<Tab>("demos");
  const [clips, setClips] = useState<DemoClip[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [clipError, setClipError] = useState<string | null>(null);
  const [clipsLoading, setClipsLoading] = useState(true);

  useEffect(() => {
    setClipsLoading(true);
    setClipError(null);
    fetch("/api/demo-clips")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setClips(data.clips || []);
        setClipsLoading(false);
      })
      .catch((err) => {
        setClipError(`Failed to load demo clips: ${err.message}`);
        setClipsLoading(false);
      });
  }, []);

  const handleClipSelect = (clip: DemoClip) => {
    const url = `/api/audio/${clip.category}/${clip.audio_file}`;
    onAudioSelect(url, clip);
  };

  const handleFileDrop = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      const url = URL.createObjectURL(file);
      onAudioSelect(url, null);
    },
    [onAudioSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      handleFileDrop(e.dataTransfer.files);
    },
    [handleFileDrop]
  );

  const TABS: { id: Tab; label: string }[] = [
    { id: "demos", label: "Demo Clips" },
    { id: "upload", label: "Upload Audio" },
  ];

  return (
    <div className="w-full">
      {/* Tabs with sliding indicator */}
      <div className="flex gap-1 mb-6 relative">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`relative px-5 py-2 text-sm rounded-md transition-colors z-10 ${
              tab === t.id
                ? "text-gold"
                : "text-warm-muted hover:text-warm-text"
            }`}
          >
            {tab === t.id && (
              <motion.div
                layoutId="tab-indicator"
                className="absolute inset-0 bg-gold/15 border border-gold/30 rounded-md"
                transition={{ type: "spring", stiffness: 400, damping: 30 }}
              />
            )}
            <span className="relative z-10">{t.label}</span>
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {tab === "demos" ? (
          <motion.div
            key="demos"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            {clipsLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="w-6 h-6 border-2 border-gold/30 border-t-gold rounded-full animate-spin" />
                <span className="ml-3 text-warm-muted text-sm">Loading clips...</span>
              </div>
            ) : clipError ? (
              <div className="border border-red-800/40 bg-red-900/20 rounded-lg p-4 text-red-400 text-sm">
                {clipError}
                <button
                  onClick={() => window.location.reload()}
                  className="ml-3 underline hover:text-red-300"
                >
                  Retry
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {clips.map((clip, i) => (
                  <motion.div
                    key={clip.id}
                    custom={i}
                    variants={cardVariants}
                    initial="hidden"
                    animate="visible"
                  >
                    <ClipCard
                      clip={clip}
                      isSelected={selectedClip?.id === clip.id}
                      onSelect={handleClipSelect}
                    />
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-all cursor-pointer ${
                isDragging
                  ? "border-gold bg-gold/5"
                  : "border-stone hover:border-warm-gray"
              }`}
              onClick={() => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = ".wav,.mp3,.flac,.ogg,.m4a";
                input.onchange = () => handleFileDrop(input.files);
                input.click();
              }}
            >
              <motion.div
                animate={isDragging ? { scale: 1.1 } : { scale: [1, 1.05, 1] }}
                transition={
                  isDragging
                    ? { duration: 0.2 }
                    : { duration: 2, repeat: Infinity, ease: "easeInOut" }
                }
                className="text-warm-gray text-4xl mb-3"
              >
                &uarr;
              </motion.div>
              <p className="text-warm-muted">
                Drag & drop audio here, or click to browse
              </p>
              <p className="text-warm-gray text-sm mt-2">
                .wav, .mp3, .flac accepted
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
