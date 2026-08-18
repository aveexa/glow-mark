"use client"

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAnalysisStore } from '@/store/analysis-store'
import { UploadDropzone } from '@/components/upload-dropzone'
import { PreviewPanel } from '@/components/preview-panel'
import { ProcessingStepper } from '@/components/processing-stepper'
import { ResultsDashboard } from '@/components/results-dashboard'
import { ErrorState } from '@/components/error-state'
import { PROCESSING_STEPS, ErrorCode, ERROR_CODES } from '@/lib/constants'
import { useToast } from '@/hooks/use-toast'
import { AnalysisResult, BackendAnalyzeApiResponse, BackendAnalyzeResponse } from '@/lib/types'
import { ImageIcon, Sparkles, Shield } from 'lucide-react'
import { saveAnalysisResult } from '@/lib/firebase/analysis'
import { useAuth } from '@/contexts/auth-context'
import { ProtectedRoute } from '@/components/protected-route'

function AnalyzePageContent() {
  const router = useRouter()
  const { toast } = useToast()
  const { user } = useAuth()
  const {
    selectedFile,
    previewUrl,
    analysisStatus,
    progressStep,
    result,
    error,
    setFile,
    setPreviewUrl,
    setAnalysisStatus,
    setProgressStep,
    setResult,
    setError,
    clearAll,
  } = useAnalysisStore()

  const [isProcessing, setIsProcessing] = useState(false)

  // Clear state on mount (no persistence)
  useEffect(() => {
    return () => {
      clearAll()
    }
  }, [clearAll])

  const handleFileSelect = (file: File) => {
    setFile(file)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    setError(null)
    setAnalysisStatus('idle')
  }

  const handleRemove = () => {
    clearAll()
  }

  const handleChange = () => {
    setFile(null)
    setPreviewUrl(null)
    setError(null)
    setAnalysisStatus('idle')
  }

  const handleCancel = () => {
    setIsProcessing(false)
    clearAll()
    router.push('/')
  }

  const processAnalysis = async () => {
    if (!selectedFile) return

    setIsProcessing(true)
    setAnalysisStatus('processing')
    setError(null)

    try {
      // Step 1: Validating
      setProgressStep(PROCESSING_STEPS[0])
      await new Promise((resolve) => setTimeout(resolve, 500))

      // Step 2: Face detection
      setProgressStep(PROCESSING_STEPS[1])
      await new Promise((resolve) => setTimeout(resolve, 200))

      // Step 4: Calculations
      setProgressStep(PROCESSING_STEPS[3])
      await new Promise((resolve) => setTimeout(resolve, 200))

      // Step 5: Score & Recommendations
      setProgressStep(PROCESSING_STEPS[4])
      await new Promise((resolve) => setTimeout(resolve, 200))

      // Call backend inference (Flask)
      const form = new FormData()
      form.append('image', selectedFile)

      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5001'
      const resp = await fetch(`${backendUrl}/analyze`, { method: 'POST', body: form })
      const data: BackendAnalyzeApiResponse = await resp.json().catch(() => ({} as any))

      if (!resp.ok) {
        const code = 'error' in data ? (data.error as ErrorCode | undefined) : undefined
        const mapped = (code && Object.values(ERROR_CODES).includes(code)) ? code : ERROR_CODES.UNKNOWN_ERROR
        setError(mapped)
        setAnalysisStatus('error')
        toast({
          title: 'Analysis Failed',
          description: mapped === ERROR_CODES.NO_FACE_DETECTED
            ? 'No face detected. Please upload a clear front-facing photo.'
            : mapped === ERROR_CODES.MULTIPLE_FACES_DETECTED
            ? 'Multiple faces detected. Please upload an image with a single face.'
            : mapped === ERROR_CODES.FILE_TOO_LARGE
            ? 'File too large. Please upload an image under 5MB.'
            : 'Could not analyze the image. Please try again.',
          variant: 'destructive',
        })
        return
      }
      const okData = data as BackendAnalyzeResponse

      // Create analysis result
      const suggestions = Array.isArray(okData?.suggestions) ? okData.suggestions : []
      const analysisResult: AnalysisResult = {
        score: Math.round(Number(okData?.score ?? 0)),
        metrics: okData?.metrics || { symmetry: 0, proportions: 0, balance: 0 },
        landmarks: Array.isArray(okData?.landmarks) ? okData.landmarks : [],
        overlayTypeHints: okData?.overlayTypeHints || { points: true, outline: true, mesh: false },
        ratios: Array.isArray(okData?.ratios) ? okData.ratios : [],
        recommendations: Array.isArray(okData?.recommendations) ? okData.recommendations : [],
        recommendation_items: Array.isArray(okData?.recommendation_items) ? okData.recommendation_items : undefined,
        suggestions,
        notes: [
          suggestions.length > 0
            ? 'Analysis computed by backend models (beauty score, geometry diagnostics, and catalog suggestions).'
            : 'Analysis computed by backend models (beauty score + recommendation model).',
          'Images are processed temporarily for inference; not stored permanently by this app.',
          'For best results, use a clear, front-facing photo with good lighting.',
        ],
      }

      setResult(analysisResult)
      setAnalysisStatus('success')

      // Save to Firestore (images are NOT saved, only analysis data)
      if (user) {
        try {
          const savedId = await saveAnalysisResult(user.uid, analysisResult)
          toast({
            title: 'Analysis Complete!',
            description: `Your aesthetic score: ${analysisResult.score}/100. Results saved to your profile.`,
          })
        } catch (saveError) {
          console.error('Failed to save analysis:', saveError)
          toast({
            title: 'Analysis Complete!',
            description: `Your aesthetic score: ${analysisResult.score}/100. Note: Failed to save to profile.`,
            variant: 'destructive',
          })
        }
      } else {
        // This shouldn't happen in /analyze since it's protected, but just in case
        toast({
          title: 'Analysis Complete!',
          description: `Your aesthetic score: ${analysisResult.score}/100`,
        })
      }
    } catch (err: any) {
      console.error('Analysis error:', err)
      
      let errorCode: ErrorCode = ERROR_CODES.UNKNOWN_ERROR
      let errorMessage = 'An unexpected error occurred'

      if (err.message?.includes('face') || err.message?.includes('landmark')) {
        errorCode = ERROR_CODES.NO_FACE_DETECTED
        errorMessage = 'Could not detect facial features. Please try a different image.'
      } else if (err.message?.includes('timeout')) {
        errorCode = ERROR_CODES.TIMEOUT
        errorMessage = 'Processing timed out. Please try again.'
      }

      setError(errorCode)
      setAnalysisStatus('error')
      toast({
        title: 'Analysis Failed',
        description: errorMessage,
        variant: 'destructive',
      })
    } finally {
      setIsProcessing(false)
      setProgressStep(null)
    }
  }

  const handleAnalyze = () => {
    processAnalysis()
  }

  const handleRetry = () => {
    if (selectedFile) {
      processAnalysis()
    } else {
      setError(null)
      setAnalysisStatus('idle')
    }
  }

  const handleAnalyzeAnother = () => {
    clearAll()
    setAnalysisStatus('idle')
  }

  const handleDelete = () => {
    clearAll()
    router.push('/')
  }

  // Render based on state
  if (error) {
    return (
      <div className="min-h-[calc(100vh-6rem)] bg-background flex items-center justify-center p-4">
        <div className="w-full max-w-2xl">
          <ErrorState errorCode={error} onRetry={handleRetry} onCancel={handleCancel} />
        </div>
      </div>
    )
  }

  if (result && analysisStatus === 'success') {
    return (
      <ResultsDashboard
        result={result}
        onAnalyzeAnother={handleAnalyzeAnother}
        onDelete={handleDelete}
      />
    )
  }

  if (isProcessing || analysisStatus === 'processing') {
    return (
      <div className="container mx-auto px-4 py-8 max-w-2xl">
        <ProcessingStepper onCancel={handleCancel} />
      </div>
    )
  }

  if (previewUrl && selectedFile) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-2xl">
        <PreviewPanel
          onAnalyze={handleAnalyze}
          onRemove={handleRemove}
          onChange={handleChange}
          isAnalyzing={isProcessing}
        />
      </div>
    )
  }

  return (
    <div className="min-h-[calc(100vh-6rem)] relative overflow-hidden bg-background">
      {/* Animated Background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-background"></div>
        <div className="absolute top-1/4 left-1/4 w-[30rem] h-[30rem] bg-primary/10 rounded-full blur-[120px] mix-blend-screen animate-pulse-slow"></div>
        <div className="absolute bottom-1/4 right-1/4 w-[30rem] h-[30rem] bg-secondary/10 rounded-full blur-[120px] mix-blend-screen"></div>
        <div className="absolute inset-0 bg-grid-gold/[0.02]"></div>
      </div>

      <div className="container mx-auto px-4 py-16 max-w-5xl relative z-10">
        {/* Header Section */}
        <div className="mb-12 text-center animate-fade-in mt-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-black/5 backdrop-blur-sm border border-black/5 text-primary text-sm font-medium mb-6 shadow-[0_0_15px_rgba(255,215,0,0.3)]">
            <Sparkles className="h-4 w-4" />
            Begin Your Consultation
          </div>
          <h1 className="text-5xl md:text-6xl font-bold mb-6 text-foreground">
            <span className="text-gradient">
              Upload Your Photo
            </span>
          </h1>
          <p className="text-xl text-muted-foreground w-full max-w-2xl mx-auto leading-relaxed font-light">
            Provide a clear, well-lit portrait to receive your personalized aesthetic and beauty analysis.
          </p>
        </div>

        {/* Upload Zone */}
        <div className="animate-slide-up">
          <UploadDropzone onFileSelect={handleFileSelect} disabled={isProcessing} />
        </div>

        {/* Info Cards */}
        <div className="grid md:grid-cols-3 gap-6 mt-12 animate-fade-in opacity-80" style={{ animationDelay: '0.2s' }}>
          <div className="bg-black/5 border border-black/5 rounded-2xl p-6 glass-panel hover:bg-black/5 transition-colors">
            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-yellow-500 to-yellow-600 flex items-center justify-center mb-4 shadow-[0_0_15px_rgba(255,215,0,0.5)]">
              <ImageIcon className="h-6 w-6 text-foreground" />
            </div>
            <h3 className="font-semibold text-foreground mb-2">High Quality Photos</h3>
            <p className="text-sm text-muted-foreground">Submit a bright, clear photo for the most accurate beauty suggestions.</p>
          </div>
          <div className="bg-black/5 border border-black/5 rounded-2xl p-6 glass-panel hover:bg-black/5 transition-colors">
            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary to-amber-600 flex items-center justify-center mb-4 shadow-[0_0_15px_rgba(255,215,0,0.5)]">
              <Sparkles className="h-6 w-6 text-foreground" />
            </div>
            <h3 className="font-semibold text-foreground mb-2">Professional Insights</h3>
            <p className="text-sm text-muted-foreground">Our intelligent system analyzes your unique proportions.</p>
          </div>
          <div className="bg-black/5 border border-black/5 rounded-2xl p-6 glass-panel hover:bg-black/5 transition-colors">
            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-yellow-500 to-yellow-600 flex items-center justify-center mb-4 shadow-[0_0_15px_rgba(255,215,0,0.5)]">
              <Shield className="h-6 w-6 text-foreground" />
            </div>
            <h3 className="font-semibold text-foreground mb-2">100% Confidential</h3>
            <p className="text-sm text-muted-foreground">Your photos are never stored and are deleted immediately after analysis.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function AnalyzePage() {
  return (
    <ProtectedRoute>
      <AnalyzePageContent />
    </ProtectedRoute>
  )
}
