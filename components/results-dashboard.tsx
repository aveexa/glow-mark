"use client"

import { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { AnalysisResult } from '@/lib/types'
import { useAnalysisStore } from '@/store/analysis-store'
import { Eye, EyeOff, Trash2, Upload, Sparkles, Activity, ShieldAlert, BadgeInfo, Box } from 'lucide-react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { FaceMesh3D } from '@/components/face-mesh-3d'

interface ResultsDashboardProps {
  result: AnalysisResult
  onAnalyzeAnother: () => void
  onDelete: () => void
}

interface ContainedImageRect {
  boxW: number
  boxH: number
  drawW: number
  drawH: number
  offsetX: number
  offsetY: number
}

/** CSS object-contain destination rect inside the laid-out image box. */
function getContainedImageRect(img: HTMLImageElement): ContainedImageRect | null {
  const boxW = img.clientWidth
  const boxH = img.clientHeight
  const nw = img.naturalWidth
  const nh = img.naturalHeight
  if (boxW <= 0 || boxH <= 0 || nw <= 0 || nh <= 0) return null

  const scale = Math.min(boxW / nw, boxH / nh)
  const drawW = nw * scale
  const drawH = nh * scale
  return {
    boxW,
    boxH,
    drawW,
    drawH,
    offsetX: (boxW - drawW) / 2,
    offsetY: (boxH - drawH) / 2,
  }
}

/** MediaPipe Face Mesh FACE_OVAL ring (ordered, closes to first). */
const FACE_OVAL_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109,
] as const

export function ResultsDashboard({
  result,
  onAnalyzeAnother,
  onDelete,
}: ResultsDashboardProps) {
  const { previewUrl } = useAnalysisStore()
  const [showOverlay, setShowOverlay] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    if (!canvasRef.current || !imageRef.current || !previewUrl) return

    const canvas = canvasRef.current
    const img = imageRef.current
    const ctx = canvas.getContext('2d')

    if (!ctx) return

    const drawOverlay = () => {
      const rect = getContainedImageRect(img)
      if (!rect) return

      canvas.width = rect.boxW
      canvas.height = rect.boxH
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (!showOverlay) return

      const toCanvas = (nx: number, ny: number) => ({
        x: rect.offsetX + nx * rect.drawW,
        y: rect.offsetY + ny * rect.drawH,
      })

      // Draw points
      if (result.overlayTypeHints.points) {
        result.landmarks.forEach((landmark) => {
          const { x, y } = toCanvas(landmark.x, landmark.y)
          ctx.fillStyle = 'rgba(168, 85, 247, 0.8)' // Primary color
          ctx.beginPath()
          ctx.arc(x, y, 4, 0, 2 * Math.PI)
          ctx.fill()

          // Add slight glow
          ctx.shadowColor = 'rgba(168, 85, 247, 1)'
          ctx.shadowBlur = 10
          ctx.fill()
          ctx.shadowBlur = 0 // Reset
        })
      }

      // Draw outline
      if (result.overlayTypeHints.outline && result.landmarks.length > 0) {
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.6)'
        ctx.lineWidth = 2
        ctx.beginPath()

        const keyPoints = FACE_OVAL_INDICES.filter((i) => i < result.landmarks.length)

        if (keyPoints.length > 1) {
          const first = result.landmarks[keyPoints[0]]
          const start = toCanvas(first.x, first.y)
          ctx.moveTo(start.x, start.y)

          for (let i = 1; i < keyPoints.length; i++) {
            const point = result.landmarks[keyPoints[i]]
            const p = toCanvas(point.x, point.y)
            ctx.lineTo(p.x, p.y)
          }
          ctx.closePath()
          ctx.stroke()
        }
      }
    }

    img.onload = drawOverlay
    if (img.complete) {
      drawOverlay()
    }

    const resizeObserver = new ResizeObserver(() => {
      drawOverlay()
    })
    resizeObserver.observe(img)

    return () => {
      resizeObserver.disconnect()
      img.onload = null
    }
  }, [result, showOverlay, previewUrl])

  const ScoreGauge = ({ score }: { score: number }) => {
    const percentage = score
    const circumference = 2 * Math.PI * 50
    const offset = circumference - (percentage / 100) * circumference
    const getScoreColor = (score: number) => {
      if (score >= 90) return 'from-yellow-400 to-yellow-500'
      if (score >= 75) return 'from-primary to-blue-500'
      if (score >= 60) return 'from-yellow-400 to-orange-500'
      return 'from-destructive to-red-600'
    }

    return (
      <div className="relative w-48 h-48 mx-auto">
        {/* Glow behind gauge */}
        <div className={cn(
          "absolute inset-0 rounded-full blur-2xl opacity-20 bg-gradient-to-tr",
          getScoreColor(score)
        )} />

        <svg className="transform -rotate-90 w-48 h-48 drop-shadow-2xl relative z-10">
          <circle
            cx="96"
            cy="96"
            r="80"
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="8"
            fill="none"
          />
          <motion.circle
            cx="96"
            cy="96"
            r="80"
            stroke="url(#gradient)"
            strokeWidth="12"
            fill="none"
            strokeDasharray={circumference * 1.6} /* Adjusted for r=80 vs r=50 */
            initial={{ strokeDashoffset: circumference * 1.6 }}
            animate={{ strokeDashoffset: offset * 1.6 }}
            transition={{ duration: 1.5, ease: "easeOut", delay: 0.5 }}
            className="drop-shadow-[0_0_15px_rgba(255,255,255,0.3)]"
            strokeLinecap="round"
          />
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={score >= 90 ? '#34d399' : score >= 75 ? 'hsl(var(--primary))' : score >= 60 ? '#fbbf24' : '#ef4444'} />
              <stop offset="100%" stopColor={score >= 90 ? '#10b981' : score >= 75 ? 'hsl(var(--secondary))' : score >= 60 ? '#f59e0b' : '#dc2626'} />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex items-center justify-center z-20">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1, duration: 0.5, type: "spring" }}
              className={`text-6xl font-black bg-gradient-to-r ${getScoreColor(score)} bg-clip-text text-transparent drop-shadow-sm`}
            >
              {score}
            </motion.div>
            <div className="text-sm font-medium text-muted-foreground mt-1 uppercase tracking-widest">Aesthetic Score</div>
          </div>
        </div>
      </div>
    )
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { type: "spring" as const, stiffness: 100 } }
  }

  return (
    <div className="w-full relative z-10 py-12">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="container mx-auto px-4 space-y-8 max-w-6xl"
      >
        {/* Header */}
        <div className="text-center mb-12">
          <motion.div variants={itemVariants} className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-black/5 border border-black/5 text-primary text-sm font-medium mb-4 backdrop-blur-md">
            <Sparkles className="h-4 w-4" />
            Analysis Complete
          </motion.div>
          <motion.h1 variants={itemVariants} className="text-4xl md:text-5xl font-extrabold mb-4 text-foreground tracking-tight">
            Your Beauty <span className="text-gradient">Profile</span>
          </motion.h1>
          <motion.p variants={itemVariants} className="text-lg text-muted-foreground max-w-2xl mx-auto font-light">
            Personalized insights on your facial harmony, proportions, and aesthetic strengths.
          </motion.p>
        </div>

        <div className="grid lg:grid-cols-12 gap-8 items-start">
          {/* Left: Image with overlay */}
          <motion.div variants={itemVariants} className="lg:col-span-5">
            <Card className="glass-panel border-black/5 shadow-2xl relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-accent/5 pointer-events-none" />
              <CardHeader className="border-b border-black/5 bg-white/[0.02]">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xl font-bold text-foreground flex items-center gap-2">
                    <Activity className="h-5 w-5 text-primary" />
                    Feature Map
                  </CardTitle>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowOverlay(!showOverlay)}
                    className="glass-panel border-black/5 hover:bg-black/5 text-xs h-8"
                  >
                    {showOverlay ? (
                      <>
                        <EyeOff className="mr-2 h-3 w-3" />
                        Hide Mesh
                      </>
                    ) : (
                      <>
                        <Eye className="mr-2 h-3 w-3" />
                        Show Mesh
                      </>
                    )}
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-0 relative">
                <div className="relative aspect-[3/4] bg-black/60 overflow-hidden flex items-center justify-center">
                  {previewUrl && (
                    <>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        ref={imageRef}
                        src={previewUrl}
                        alt="Analysis visualization"
                        className="w-full h-full object-contain filter contrast-[1.05]"
                      />
                      <canvas
                        ref={canvasRef}
                        className="absolute inset-0 pointer-events-none transition-opacity duration-300"
                        style={{ opacity: showOverlay ? 1 : 0 }}
                      />
                    </>
                  )}
                  {/* Subtle vignette */}
                  <div className="absolute inset-0 bg-radial-gradient from-transparent to-black/40 pointer-events-none" />
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Right: Score and Metrics */}
          <div className="lg:col-span-7 space-y-8">
            <motion.div variants={itemVariants}>
              <Card className="glass-panel border-black/5 shadow-[0_0_40px_-10px_rgba(255,215,0,0.2)] relative overflow-hidden">
                <div className="absolute inset-0 bg-grid-gold/[0.02] pointer-events-none" />
                <CardHeader className="pb-2 border-b border-black/5 bg-white/[0.02]">
                  <CardTitle className="text-xl font-bold text-foreground text-center tracking-wide">Overall Rating</CardTitle>
                </CardHeader>
                <CardContent className="pt-8 pb-10">
                  <ScoreGauge score={result.score} />
                </CardContent>
              </Card>
            </motion.div>

            <motion.div variants={itemVariants} className="grid grid-cols-3 gap-4">
              {[
                { label: 'Symmetry', value: result.metrics.symmetry, color: 'from-yellow-400 to-amber-500' },
                { label: 'Proportions', value: result.metrics.proportions, color: 'from-amber-400 to-yellow-500' },
                { label: 'Balance', value: result.metrics.balance, color: 'from-yellow-400 to-amber-500' }
              ].map((metric, idx) => (
                <Card key={idx} className="glass-panel border-black/5 hover:border-white/20 transition-all duration-300 group overflow-hidden relative">
                  <div className={`absolute inset-0 bg-gradient-to-br ${metric.color} opacity-0 group-hover:opacity-10 transition-opacity duration-500`} />
                  <CardContent className="pt-6 pb-6 text-center relative z-10">
                    <div className={`text-4xl font-extrabold bg-gradient-to-r ${metric.color} bg-clip-text text-transparent mb-2 drop-shadow-md`}>
                      {metric.value}
                    </div>
                    <div className="text-xs font-medium text-muted-foreground uppercase tracking-widest">{metric.label}</div>
                  </CardContent>
                </Card>
              ))}
            </motion.div>

            {/* Actions (Desktop) */}
            <motion.div variants={itemVariants} className="hidden lg:flex gap-4">
              <Button
                onClick={onAnalyzeAnother}
                className="flex-1 bg-gradient-to-r from-primary to-accent hover:opacity-90 shadow-[0_0_20px_-5px_hsl(270,100%,60%,0.4)] transition-all duration-300 h-14 text-lg font-semibold rounded-xl"
              >
                <Upload className="mr-2 h-5 w-5" />
                New Evaluation
              </Button>
              <Button
                onClick={onDelete}
                variant="destructive"
                className="hover:bg-red-900/50 bg-destructive/20 text-red-500 hover:text-red-400 border border-destructive/30 transition-all duration-300 h-14 text-lg font-semibold rounded-xl px-8"
              >
                <Trash2 className="mr-2 h-5 w-5" />
                Clear Results
              </Button>
            </motion.div>
          </div>
        </div>

        {/* Tabs */}
        <motion.div variants={itemVariants}>
          <Card className="glass-panel border-black/5 shadow-2xl">
            <CardContent className="p-0">
              <Tabs defaultValue="insights" className="w-full">
                <TabsList className="w-full flex border-b border-black/5 bg-white/[0.02] p-0 h-14 rounded-t-xl overflow-x-auto overflow-y-hidden custom-scrollbar">
                  {['Insights', 'Ratios', 'Recommendations', '3D View', 'Notes'].map((tab) => (
                    <TabsTrigger
                      key={tab.toLowerCase().replace(' ', '-')}
                      value={tab.toLowerCase().replace(' ', '-')}
                      className="flex-1 data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:text-primary text-muted-foreground rounded-none h-full transition-all text-sm font-semibold tracking-wide uppercase"
                    >
                      {tab === '3D View' ? (
                        <span className="flex items-center gap-1">
                          <Box className="h-3 w-3" />
                          3D View
                        </span>
                      ) : (
                        tab
                      )}
                    </TabsTrigger>
                  ))}
                </TabsList>

                <div className="p-6 md:p-8">
                  <TabsContent value="insights" className="mt-0 focus-visible:outline-none focus-visible:ring-0">
                    <div className="space-y-4 p-6 bg-primary/5 rounded-2xl border border-primary/20 relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-primary/20 blur-[50px] rounded-full pointer-events-none" />
                      <div className="flex items-start gap-4 relative z-10">
                        <BadgeInfo className="h-6 w-6 text-primary flex-shrink-0 mt-1" />
                        <p className="text-lg text-foreground leading-relaxed font-light">
                          Your facial analysis shows a beautiful profile with excellent symmetry
                          and proportional features. Our experts have mapped {result.landmarks.length} key points
                          and calculated comprehensive metrics across multiple dimensions to generate this personalized profile.
                        </p>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="ratios" className="mt-0 focus-visible:outline-none focus-visible:ring-0">
                    <div className="grid md:grid-cols-2 gap-4">
                      {result.ratios.map((ratio, index) => (
                        <div key={index} className="p-5 glass-panel border-black/5 hover:border-black/5 rounded-2xl group transition-all">
                          <div className="flex justify-between items-center mb-3">
                            <span className="font-semibold text-foreground">{ratio.name}</span>
                            <span className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-primary to-accent">{ratio.value.toFixed(2)}</span>
                          </div>
                          <div className="flex items-center gap-4">
                            <div className="flex-1 h-2.5 bg-black/40 rounded-full overflow-hidden border border-black/5 relative">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.min((ratio.value / 2) * 100, 100)}%` }}
                                transition={{ duration: 1, delay: 0.2 + index * 0.1 }}
                                className="absolute top-0 left-0 h-full bg-gradient-to-r from-primary to-accent shadow-[0_0_10px_rgba(255,215,0,0.5)]"
                              />
                            </div>
                            <span className="text-xs text-muted-foreground font-medium min-w-[80px] text-right font-mono">
                              IDL: {ratio.idealRange}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </TabsContent>

                  <TabsContent value="recommendations" className="mt-0 focus-visible:outline-none focus-visible:ring-0">
                    <div className="space-y-8">
                      <div>
                        <h3 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground mb-4">
                          Personalized tips
                        </h3>
                        {(result.suggestions?.length ?? 0) > 0 ? (
                          <ul className="space-y-4">
                            {result.suggestions!.map((sug, index) => (
                              <motion.li
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                                key={sug.id || index}
                                className="flex items-start gap-4 p-5 glass-panel border-black/5 hover:border-primary/20 rounded-2xl transition-all group"
                              >
                                <div className="h-8 w-8 rounded-full bg-primary/10 border border-primary/30 flex items-center justify-center flex-shrink-0 mt-0.5 group-hover:bg-primary/20 transition-colors">
                                  <span className="text-primary text-sm font-bold">{index + 1}</span>
                                </div>
                                <div className="flex-1 min-w-0">
                                  <span className="text-lg text-foreground leading-relaxed font-light block">
                                    {sug.text}
                                  </span>
                                  {typeof sug.confidence === 'number' && (
                                    <span className="text-xs text-muted-foreground mt-2 inline-block font-mono">
                                      Confidence {(sug.confidence * 100).toFixed(0)}%
                                    </span>
                                  )}
                                </div>
                              </motion.li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-sm text-muted-foreground leading-relaxed p-4 rounded-xl bg-black/[0.02] border border-black/5">
                            Catalog tips are unavailable for this analysis. Geometry diagnostics below still reflect the recommendation model.
                          </p>
                        )}
                      </div>

                      {(() => {
                        const items =
                          result.recommendation_items?.filter((it) => it.class !== 'ok') ??
                          []
                        const diagnosticRows =
                          items.length > 0
                            ? items.map(
                                (it) =>
                                  `${it.label}: ${it.class}${
                                    typeof it.confidence === 'number'
                                      ? ` (${(it.confidence * 100).toFixed(0)}%)`
                                      : ''
                                  }`
                              )
                            : result.recommendations
                        if (diagnosticRows.length === 0) return null
                        return (
                          <div>
                            <h3 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground mb-4">
                              Geometry diagnostics
                            </h3>
                            <ul className="space-y-3">
                              {diagnosticRows.map((rec, index) => (
                                <motion.li
                                  initial={{ opacity: 0, x: -12 }}
                                  animate={{ opacity: 1, x: 0 }}
                                  transition={{ delay: 0.05 * index }}
                                  key={`diag-${index}`}
                                  className="flex items-start gap-3 p-4 rounded-xl border border-black/5 bg-black/[0.02]"
                                >
                                  <Activity className="h-4 w-4 text-muted-foreground flex-shrink-0 mt-1" />
                                  <span className="text-sm text-foreground/90 leading-relaxed font-mono">
                                    {rec}
                                  </span>
                                </motion.li>
                              ))}
                            </ul>
                          </div>
                        )
                      })()}
                    </div>
                  </TabsContent>

                  <TabsContent value="3d-view" className="mt-0 focus-visible:outline-none focus-visible:ring-0">
                    <div className="space-y-4">
                      <div className="p-6 bg-primary/5 rounded-2xl border border-primary/20">
                        <div className="flex items-start gap-4 mb-4">
                          <Box className="h-6 w-6 text-primary flex-shrink-0 mt-1" />
                          <div>
                            <h3 className="text-lg font-semibold text-foreground mb-2">3D Facial Mesh Visualization</h3>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              Interactive 3D representation of your facial landmarks. Rotate, zoom, and explore your facial structure from all angles.
                            </p>
                          </div>
                        </div>
                        <FaceMesh3D landmarks={result.landmarks} />
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="notes" className="mt-0 focus-visible:outline-none focus-visible:ring-0">
                    <div className="space-y-4">
                      {result.notes.map((note, index) => (
                        <div key={index} className="p-5 bg-yellow-500/5 border border-yellow-500/20 rounded-2xl flex items-start gap-4">
                          <ShieldAlert className="h-5 w-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                          <p className="text-base text-foreground leading-relaxed font-light">
                            {note}
                          </p>
                        </div>
                      ))}
                    </div>
                  </TabsContent>
                </div>
              </Tabs>
            </CardContent>
          </Card>
        </motion.div>

        {/* Actions (Mobile) */}
        <motion.div variants={itemVariants} className="flex flex-col gap-4 lg:hidden">
          <Button
            onClick={onAnalyzeAnother}
            className="w-full bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-all duration-300 h-14 text-lg font-semibold rounded-xl"
          >
            <Upload className="mr-2 h-5 w-5" />
            New Evaluation
          </Button>
          <Button
            onClick={onDelete}
            variant="destructive"
            className="w-full hover:bg-red-900/50 bg-destructive/20 text-red-500 hover:text-red-400 border border-destructive/30 transition-all duration-300 h-14 text-lg font-semibold rounded-xl"
          >
            <Trash2 className="mr-2 h-5 w-5" />
            Clear Results
          </Button>
        </motion.div>

        {/* Disclaimer */}
        <motion.div variants={itemVariants} className="text-sm text-muted-foreground text-center p-6 glass-panel rounded-2xl border-black/5 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/[0.02] to-transparent pointer-events-none" />
          <p className="font-semibold text-foreground mb-2 flex items-center justify-center gap-2 tracking-wide uppercase">
            <ShieldAlert className="h-4 w-4 text-yellow-500" />
            Analysis Disclaimer
          </p>
          <p className="leading-relaxed font-light max-w-4xl mx-auto">
            This analysis is generated dynamically for <strong className="text-foreground">entertainment and educational purposes only</strong>. Results are based on
            general aesthetic principles and automated estimates. They should <strong className="text-foreground">not be interpreted as professional medical or cosmetic advice</strong>.
            Your photos are processed securely and <strong className="text-foreground">temporarily, deleted immediately post-analysis</strong>.
          </p>
        </motion.div>
      </motion.div>
    </div>
  )
}
