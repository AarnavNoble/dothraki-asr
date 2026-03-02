"use client";

import { motion } from "framer-motion";
import GoldDivider from "./ui/GoldDivider";

const TECH = [
  "Whisper",
  "PyTorch",
  "FastAPI",
  "Next.js",
  "Tailwind CSS",
  "wavesurfer.js",
];

const badgeVariants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: (i: number) => ({
    opacity: 1,
    scale: 1,
    transition: { delay: 0.2 + i * 0.08, duration: 0.3 },
  }),
};

export default function About() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="pt-16 pb-12 px-4"
    >
      <GoldDivider className="max-w-lg mx-auto mb-12" />
      <div className="max-w-2xl mx-auto text-center space-y-4">
        {/* Heading with flanking gold lines */}
        <div className="flex items-center justify-center gap-4">
          <div className="h-px w-12 bg-gradient-to-r from-transparent to-gold/40" />
          <h2 className="font-serif text-2xl text-gold">About</h2>
          <div className="h-px w-12 bg-gradient-to-l from-transparent to-gold/40" />
        </div>

        <p className="text-warm-muted leading-relaxed text-sm">
          Dothraki ASR is a multi-strategy automatic speech recognition pipeline
          built for the constructed Dothraki language from Game of Thrones. It
          combines phoneme matching, embedding similarity, dynamic time warping,
          and a fine-tuned Whisper decoder &mdash; with an ensemble mode that merges
          all strategies for the best results.
        </p>
        <p className="text-warm-muted leading-relaxed text-sm">
          The system was trained on 1,700+ synthetic audio clips with IPA
          transcriptions, covering dialogue from all eight seasons.
        </p>

        {/* Tech badges stagger in */}
        <div className="flex flex-wrap justify-center gap-2 pt-4">
          {TECH.map((t, i) => (
            <motion.span
              key={t}
              custom={i}
              variants={badgeVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="text-xs text-warm-gray border border-stone px-3 py-1 rounded-full hover:border-gold/30 hover:text-warm-muted transition-colors"
            >
              {t}
            </motion.span>
          ))}
        </div>
      </div>
    </motion.section>
  );
}
