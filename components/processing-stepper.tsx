"use client"

import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { PROCESSING_STEPS } from '@/lib/constants'
import { useAnalysisStore } from '@/store/analysis-store'
import { Check, Loader2, Sparkles, Activity } from 'lucide-react'
import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'
import { motion, AnimatePresence } from 'framer-motion'

interface ProcessingStepperProps {
  onCancel: () => void
}

export function ProcessingStepper({ onCancel }: ProcessingStepperProps) {
  const { progressStep } = useAnalysisStore()
  const [currentStepIndex, setCurrentStepIndex] = useState(0)

  useEffect(() => {
    if (progressStep) {
      const index = PROCESSING_STEPS.indexOf(progressStep)
      if (index !== -1) {
        setCurrentStepIndex(index)
      }
    }
  }, [progressStep])

  const progress = ((currentStepIndex + 1) / PROCESSING_STEPS.length) * 100

  return (
    <div className="w-full relative z-10 py-12">
      <motion.div
        initial={{ opacity: 0, y: 30, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ type: "spring", stiffness: 100, damping: 20 }}
        className="container mx-auto px-4 flex justify-center"
      >
        <Card className="w-full max-w-2xl glass-panel border-black/5 shadow-[0_0_50px_-12px_rgba(255,215,0,0.3)] relative overflow-hidden rounded-[2rem]">
          {/* Animated Background Gradient */}
          <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-accent/10 opacity-70 pointer-events-none" />

          <CardContent className="p-8 md:p-12 relative z-10">
            {/* Header */}
            <div className="text-center mb-10">
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-black/5 border border-black/5 text-primary text-sm font-medium mb-4"
              >
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing Features
              </motion.div>
              <h2 className="text-3xl md:text-4xl font-extrabold text-foreground mb-3 tracking-tight">Generating Profile</h2>
              <p className="text-muted-foreground font-light text-lg">Evaluating facial proportions and discovering beauty insights...</p>
            </div>

            <div className="space-y-4 mb-10 relative">
              {/* Connecting Line background */}
              <div className="absolute left-[2.25rem] top-8 bottom-8 w-px bg-black/5" />

              <AnimatePresence>
                {PROCESSING_STEPS.map((step, index) => {
                  const isActive = index === currentStepIndex
                  const isCompleted = index < currentStepIndex
                  const isPending = !isActive && !isCompleted

                  return (
                    <motion.div
                      key={step}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className={cn(
                        "relative flex items-center gap-6 p-4 rounded-2xl transition-all duration-500",
                        isActive ? "bg-black/5 border border-primary/30 shadow-[0_0_30px_-5px_rgba(255,215,0,0.3)] z-10 scale-[1.02]" : "border border-transparent hover:bg-white/[0.02]",
                        isCompleted && "bg-transparent",
                        isPending && "opacity-50"
                      )}
                    >
                      <div className="flex-shrink-0 relative z-10 bg-background rounded-full">
                        {isCompleted ? (
                          <motion.div
                            initial={{ scale: 0 }} animate={{ scale: 1 }}
                            className="h-14 w-14 rounded-full bg-gradient-to-br from-yellow-500 to-yellow-600 flex items-center justify-center shadow-[0_0_20px_rgba(255,215,0,0.4)]"
                          >
                            <Check className="h-6 w-6 text-foreground" />
                          </motion.div>
                        ) : isActive ? (
                          <div className="relative z-10">
                            <div className="h-14 w-14 rounded-full bg-gradient-to-br from-primary via-amber-500 to-accent flex items-center justify-center shadow-[0_0_20px_rgba(255,215,0,0.6)] animate-pulse relative">
                              <Activity className="h-6 w-6 text-foreground animate-pulse absolute" />
                              <div className="absolute inset-0 bg-white/20 rounded-full blur-md opacity-50" />
                            </div>
                          </div>
                        ) : (
                          <div className="h-14 w-14 rounded-full border border-white/20 bg-black/5 flex items-center justify-center backdrop-blur-sm">
                            <span className="text-lg font-medium text-foreground/40">{index + 1}</span>
                          </div>
                        )}
                      </div>
                      <div className="flex-1">
                        <motion.p
                          animate={{ color: isActive ? "#fff" : isCompleted ? "#a7f3d0" : "#64748b" }}
                          className="text-lg font-medium tracking-wide"
                        >
                          {step}
                        </motion.p>
                      </div>
                    </motion.div>
                  )
                })}
              </AnimatePresence>
            </div>

            {/* Progress Bar Area */}
            <div className="space-y-4 bg-black/5 p-6 rounded-2xl border border-black/5 backdrop-blur-md relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-transparent opacity-50 pointer-events-none" />
              <div className="relative flex justify-between items-center z-10">
                <span className="text-sm font-medium text-muted-foreground uppercase tracking-widest">Compute Progress</span>
                <motion.span
                  key={Math.round(progress)}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-base font-bold text-primary"
                >
                  {Math.round(progress)}%
                </motion.span>
              </div>

              <div className="relative h-2 w-full bg-black/5 rounded-full overflow-hidden">
                <motion.div
                  className="absolute top-0 left-0 h-full bg-gradient-to-r from-primary via-amber-500 to-accent shadow-[0_0_10px_rgba(255,215,0,0.5)]"
                  initial={{ width: "0%" }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                />
              </div>
            </div>

            {/* Cancel Action */}
            <div className="flex justify-center pt-8">
              <button
                onClick={onCancel}
                className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors uppercase tracking-widest py-2 px-6 rounded-full hover:bg-black/5"
              >
                Cancel Analysis
              </button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
