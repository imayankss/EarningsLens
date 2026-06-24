"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type SectionShellProps = {
  eyebrow: string;
  title: string;
  description: string;
  children: ReactNode;
  className?: string;
};

export function SectionShell({
  eyebrow,
  title,
  description,
  children,
  className,
}: SectionShellProps) {
  return (
    <section className={cn("relative mx-auto w-full max-w-7xl px-4 py-16 sm:px-6 lg:px-8", className)}>
      <motion.div
        initial={{ opacity: 0, y: 22 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.55, ease: "easeOut" }}
        className="mb-8 max-w-3xl"
      >
        <Badge
          variant="outline"
          className="mb-4 border-cyan-400/25 bg-cyan-400/10 font-mono text-cyan-200"
        >
          {eyebrow}
        </Badge>
        <h2 className="text-3xl font-semibold text-white sm:text-4xl">
          {title}
        </h2>
        <p className="mt-4 text-base leading-7 text-slate-300 sm:text-lg">
          {description}
        </p>
      </motion.div>
      {children}
    </section>
  );
}
