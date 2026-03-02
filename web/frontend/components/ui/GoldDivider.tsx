"use client";

import { motion } from "framer-motion";

export default function GoldDivider({ className = "" }: { className?: string }) {
  return (
    <div className={`relative ${className}`}>
      <motion.div
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 1.2, ease: "easeOut" }}
        className="h-px bg-gradient-to-r from-transparent via-gold/60 to-transparent"
      />
      {/* Shimmer sweep overlay */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2, duration: 0.3 }}
        className="absolute inset-0 h-px shimmer-border"
        style={{ boxShadow: "0 0 8px rgba(200,164,92,0.2)" }}
      />
    </div>
  );
}
