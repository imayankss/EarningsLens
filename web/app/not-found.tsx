import Link from "next/link";
import { ArrowLeft, SearchX } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[#050816] px-4">
      <Card className="w-full max-w-xl text-center">
        <CardContent className="p-8 sm:p-10">
          <SearchX aria-hidden className="mx-auto size-10 text-cyan-200" />
          <p className="mt-5 font-mono text-sm text-cyan-200">404 · NOT FOUND</p>
          <h1 className="mt-3 text-3xl font-semibold text-white">That research view does not exist.</h1>
          <p className="mt-4 text-sm leading-6 text-slate-400">Return to the verified EarningsLens dashboard or review the published methodology.</p>
          <div className="mt-7 flex flex-wrap justify-center gap-3">
            <Button asChild>
              <Link href="/"><ArrowLeft aria-hidden className="size-4" />Dashboard</Link>
            </Button>
            <Button asChild variant="outline" className="border-white/15 bg-white/5">
              <Link href="/methodology">Methodology</Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}
