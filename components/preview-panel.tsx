"use client"

import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Image, X, RotateCcw, Activity, Sparkles } from 'lucide-react'
import { useAnalysisStore } from '@/store/analysis-store'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'

interface PreviewPanelProps {
  onAnalyze: () => void
  onRemove: () => void
  onChange: () => void
  isAnalyzing?: boolean
}

export function PreviewPanel({
  onAnalyze,
  onRemove,
  onChange,
  isAnalyzing = false,
}: PreviewPanelProps) {
  const { previewUrl, selectedFile } = useAnalysisStore()

  if (!previewUrl || !selectedFile) {
    return null
  }

  return (
    <div className="w-full relative z-10 py-8">
      <motion.div
        initial={{ opacity: 0, y: 30, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ type: "spring", stiffness: 100, damping: 15 }}
        className="container mx-auto px-4 max-w-4xl"
      >
        <Card className="glass-panel border-black/5 overflow-hidden relative shadow-2xl">
          {/* Animated glow behind card */}
          <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-accent/10 opacity-50 pointer-events-none" />

          <CardContent className="p-8 md:p-12 relative z-10">
            <div className="space-y-10">

              {/* Header */}
              <div className="text-center">
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 }}
                  className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-black/5 border border-black/5 text-primary text-sm font-medium mb-4 backdrop-blur-md"
                >
                  <Image className="h-4 w-4" />
                  Photo Uploaded
                </motion.div>
                <h2 className="text-3xl md:text-4xl font-extrabold text-foreground mb-3 tracking-tight">Ready for Review</h2>
                <p className="text-muted-foreground font-light text-lg">Verify image clarity before generating your beauty profile.</p>
              </div>

              {/* Image Preview Area */}
              <div className="relative max-w-2xl mx-auto group">
                {/* Outer Glow */}
                <div className={cn(
                  "absolute -inset-4 rounded-3xl blur-2xl transition-all duration-1000",
                  isAnalyzing ? "bg-primary/40 animate-pulse-slow" : "bg-black/5 group-hover:bg-primary/20"
                )} />

                <div className="relative aspect-[4/3] rounded-2xl overflow-hidden bg-black/40 border border-black/5 shadow-2xl flex items-center justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl}
                    alt="Facial Subject Preview"
                    className={cn(
                      "w-full h-full object-contain transition-all duration-700",
                      isAnalyzing && "scale-105 filter contrast-125 brightness-90 saturate-50"
                    )}
                  />

                  {/* Overlay Vignette */}
                  <div className="absolute inset-0 bg-radial-gradient from-transparent to-black/60 pointer-events-none" />

                  {/* Scanning Animation */}
                  <AnimatePresence>
                    {isAnalyzing && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 pointer-events-none overflow-hidden"
                      >
                        {/* Scanning Line */}
                        <motion.div
                          animate={{ y: ["0%", "100%", "0%"] }}
                          transition={{ duration: 3, ease: "linear", repeat: Infinity }}
                          className="absolute w-full h-1 bg-primary left-0 shadow-[0_0_20px_10px_rgba(255,215,0,0.5)]"
                          style={{ top: "0%" }}
                        />
                        {/* Grid overlay when analyzing */}
                        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCI+PHBhdGggZD0iTTAgMGgyMHYyMEgwem0xIDE5aDE4VjFIMXoiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4wNSkiIHJ1bGU9ImV2ZW5vZGQiLz48L3N2Zz4=')] opacity-20 MixBlendMode-overlay" />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center max-w-2xl mx-auto mt-8">
                <Button
                  onClick={onAnalyze}
                  disabled={isAnalyzing}
                  className="flex-1 group relative overflow-hidden bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-all duration-300 transform shadow-[0_0_30px_-5px_hsl(270,100%,60%,0.4)] rounded-xl border border-black/5 h-14 text-lg"
                >
                  <span className="relative z-10 flex items-center justify-center font-semibold tracking-wide text-black">
                    {isAnalyzing ? (
                      <>
                        <Activity className="animate-pulse mr-2 h-5 w-5" />
                        Analyzing Features...
                      </>
                    ) : (
                      <>
                        <Sparkles className="mr-2 h-5 w-5" />
                        Start Analysis
                      </>
                    )}
                  </span>
                  {!isAnalyzing && (
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
                  )}
                </Button>

                <div className="flex gap-4 sm:flex-none">
                  <Button
                    onClick={onChange}
                    disabled={isAnalyzing}
                    variant="outline"
                    className="flex-1 sm:flex-none sm:w-auto glass-panel border-black/5 hover:bg-black/5 hover:text-foreground transition-all duration-300 h-14 rounded-xl px-6"
                  >
                    <RotateCcw className="mr-2 h-5 w-5 text-muted-foreground" />
                    Reset
                  </Button>

                  <Button
                    onClick={onRemove}
                    disabled={isAnalyzing}
                    variant="destructive"
                    className="flex-1 sm:flex-none sm:w-auto hover:bg-red-900/50 bg-destructive/20 text-red-500 hover:text-red-400 border border-destructive/30 shadow-lg transition-all duration-300 h-14 rounded-xl px-6"
                  >
                    <X className="mr-2 h-5 w-5" />
                    Discard
                  </Button>
                </div>
              </div>

            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
