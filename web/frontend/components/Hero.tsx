"use client";

import { motion } from "framer-motion";
import GoldDivider from "./ui/GoldDivider";

const EMBERS = [
  { left: "10%", delay: 0, duration: 12 },
  { left: "25%", delay: 3, duration: 15 },
  { left: "45%", delay: 1, duration: 10 },
  { left: "65%", delay: 5, duration: 14 },
  { left: "80%", delay: 2, duration: 11 },
  { left: "92%", delay: 4, duration: 13 },
];

const TITLE = "Dothraki ASR";

const letterVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.3 + i * 0.06, duration: 0.4, ease: "easeOut" as const },
  }),
};

export default function Hero() {
  return (
    <section className="relative grain-overlay flex flex-col items-center justify-center pt-24 pb-16 px-4 overflow-hidden">
      {/* Floating embers */}
      {EMBERS.map((e, i) => (
        <div
          key={i}
          className="ember"
          style={{
            left: e.left,
            bottom: "-10px",
            animationDelay: `${e.delay}s`,
            animationDuration: `${e.duration}s`,
          }}
        />
      ))}

      {/* Radial gold gradient orb behind title */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[300px] rounded-full pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse, rgba(200,164,92,0.08) 0%, transparent 70%)",
        }}
      />

      {/* Title: staggered letter-by-letter reveal */}
      <h1 className="font-serif text-5xl md:text-7xl text-gold animate-glow-pulse tracking-wide text-center relative z-10 flex">
        {TITLE.split("").map((char, i) => (
          <motion.span
            key={i}
            custom={i}
            variants={letterVariants}
            initial="hidden"
            animate="visible"
            className={char === " " ? "inline-block w-4" : "inline-block"}
          >
            {char}
          </motion.span>
        ))}
      </h1>

      {/* Subtitle: fades + slides up after title completes */}
      <motion.p
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 1.2 }}
        className="mt-4 text-warm-muted text-lg md:text-xl text-center relative z-10"
      >
        Transcribing the language of the Great Grass Sea
      </motion.p>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.6, duration: 0.5 }}
        className="w-full max-w-md"
      >
        <GoldDivider className="mt-10 w-full" />
      </motion.div>
    </section>
  );
}
