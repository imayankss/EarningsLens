"use client";

import { TrendingDown, TrendingUp } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { TickerMovement } from "@/types/dashboard";

const toneStyles = {
  positive: "text-emerald-300 border-emerald-400/30 bg-emerald-400/10",
  negative: "text-rose-300 border-rose-400/30 bg-rose-400/10",
  neutral: "text-slate-300 border-slate-400/20 bg-slate-400/10",
  warning: "text-amber-300 border-amber-400/30 bg-amber-400/10",
};

export function FinanceTicker({ items }: { items: TickerMovement[] }) {
  const tickerItems = [...items, ...items, ...items, ...items];

  return (
    <div className="relative overflow-hidden border-y border-white/10 bg-slate-950/72 py-3 backdrop-blur-xl">
      <div className="ticker-track flex w-max gap-3 px-3">
        {tickerItems.map((item, index) => {
          const DirectionIcon = item.tone === "negative" ? TrendingDown : TrendingUp;

          return (
            <div
              key={`${item.symbol}-${index}`}
              className="flex min-w-64 items-center justify-between gap-4 rounded-md border border-white/10 bg-white/[0.04] px-4 py-3 font-mono shadow-glow"
            >
              <div>
                <div className="text-sm font-semibold text-white">{item.symbol}</div>
                <div className="text-xs text-slate-400">{item.company}</div>
              </div>
              <Badge variant="outline" className={toneStyles[item.tone]}>
                {item.sentiment}
              </Badge>
              <div className={`flex items-center gap-1 text-sm ${item.tone === "negative" ? "text-rose-300" : "text-emerald-300"}`}>
                <DirectionIcon className="size-4" />
                {item.move}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
