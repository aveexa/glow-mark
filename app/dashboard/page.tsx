"use client"

import { Sparkles, History, User, Activity, ArrowRight, ShieldCheck, LogOut, Loader2, ImageIcon, Shield } from "lucide-react"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { motion } from "framer-motion"
import { ProtectedRoute } from "@/components/protected-route"
import { useAuth } from "@/contexts/auth-context"
import { getUserAnalyses, getUserAverageMetrics, AnalysisRecord, saveAnalysisResult } from "@/lib/firebase/analysis"
import { useEffect, useState, useCallback } from "react"
import { useToast } from "@/hooks/use-toast"
import { useAnalysisStore } from "@/store/analysis-store"
import { UploadDropzone } from "@/components/upload-dropzone"
import { PreviewPanel } from "@/components/preview-panel"
import { ProcessingStepper } from "@/components/processing-stepper"
import { ResultsDashboard } from "@/components/results-dashboard"
import { ErrorState } from "@/components/error-state"
import { PROCESSING_STEPS, ErrorCode, ERROR_CODES } from "@/lib/constants"
import { AnalysisResult, BackendAnalyzeResponse, BackendAnalyzeApiResponse } from "@/lib/types"
import { signOutUser } from "@/lib/firebase/auth"

function DashboardContent() {
    const { user, userData } = useAuth()
    const { toast } = useToast()
    const [analyses, setAnalyses] = useState<AnalysisRecord[]>([])
    const [averageMetrics, setAverageMetrics] = useState({
        averageScore: 0,
        averageSymmetry: 0,
        averageProportions: 0,
        averageBalance: 0,
        totalAnalyses: 0,
    })
    const [loading, setLoading] = useState(true)
    const [showAnalysis, setShowAnalysis] = useState(false)
    
    // Analysis state
    const {
        selectedFile,
        previewUrl,
        analysisStatus,
        progressStep,
        result,
        error,
        errorHint,
        setFile,
        setPreviewUrl,
        setAnalysisStatus,
        setProgressStep,
        setResult,
        setError,
        clearAll,
    } = useAnalysisStore()
    const [isProcessing, setIsProcessing] = useState(false)
    const [regionPending, setRegionPending] = useState(false)

    useEffect(() => {
        if (user) {
            loadAnalyses()
        }
    }, [user])

    // Initialize MediaPipe on component mount
    useEffect(() => {
        if (showAnalysis) {
            import('@/lib/mediapipe/face-landmarker').then(({ initializeFaceLandmarker }) => {
                initializeFaceLandmarker().catch(console.error)
            })
        }
    }, [showAnalysis])

    const loadAnalyses = async () => {
        if (!user) return
        
        setLoading(true)
        try {
            const [userAnalyses, metrics] = await Promise.all([
                getUserAnalyses(user.uid),
                getUserAverageMetrics(user.uid)
            ])
            
            setAnalyses(userAnalyses)
            setAverageMetrics(metrics)
        } catch (error) {
            console.error('Error loading analyses:', error)
            toast({
                title: 'Error',
                description: 'Failed to load consultation history',
                variant: 'destructive',
            })
        } finally {
            setLoading(false)
        }
    }

    const formatDate = (date: Date) => {
        return new Intl.DateTimeFormat('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
        }).format(date)
    }

    const getTopTraits = (metrics: { symmetry: number; proportions: number; balance: number }) => {
        const traits: string[] = []
        if (metrics.symmetry >= 85) traits.push('High Symmetry')
        if (metrics.proportions >= 85) traits.push('Excellent Proportions')
        if (metrics.balance >= 85) traits.push('Perfect Balance')
        if (metrics.symmetry >= 80 && metrics.symmetry < 85) traits.push('Good Symmetry')
        if (metrics.proportions >= 80 && metrics.proportions < 85) traits.push('Balanced Features')
        if (traits.length === 0) traits.push('Analyzing...')
        return traits.slice(0, 2)
    }

    const handleStartAnalysis = () => {
        setShowAnalysis(true)
        clearAll()
    }

    const handleFileSelect = useCallback((file: File) => {
        setFile(file)
        const url = URL.createObjectURL(file)
        setPreviewUrl(url)
        setError(null)
        setAnalysisStatus('idle')
    }, [setFile, setPreviewUrl, setError, setAnalysisStatus])

    const handleRemove = () => {
        clearAll()
        setShowAnalysis(false)
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
        setShowAnalysis(false)
    }

    // regionOverride re-runs the same photo against a different comparison group.
    // Session-only: it is sent per request, never persisted, and an override re-run
    // does not create a second saved analysis.
    const processAnalysis = async (regionOverride?: string) => {
        if (!selectedFile || !user) return
        const isRegionRerun = regionOverride !== undefined

        // A region re-run keeps the results on screen with an inline pending marker;
        // swapping to the progress view for what is a re-read of the same photo would
        // read as losing the result. The staged delays below are presentation for a
        // first analysis and are skipped here for the same reason.
        if (isRegionRerun) {
            setRegionPending(true)
        } else {
            setIsProcessing(true)
            setAnalysisStatus('processing')
        }
        setError(null)

        try {
            if (!isRegionRerun) {
                setProgressStep(PROCESSING_STEPS[0])
                await new Promise((resolve) => setTimeout(resolve, 500))
                setProgressStep(PROCESSING_STEPS[1])
                await new Promise((resolve) => setTimeout(resolve, 300))
                setProgressStep(PROCESSING_STEPS[2])
            }
            // Call the backend inference (orchestrator) — same path as the /analyze page
            const form = new FormData()
            form.append('image', selectedFile)
            if (regionOverride !== undefined) form.append('region_override', regionOverride)
            const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5001'
            const resp = await fetch(`${backendUrl}/analyze`, { method: 'POST', body: form })
            const data: BackendAnalyzeApiResponse = await resp.json().catch(() => ({} as any))

            if (!resp.ok) {
                const code = 'error' in data ? (data.error as ErrorCode | undefined) : undefined
                const mapped = (code && Object.values(ERROR_CODES).includes(code)) ? code : ERROR_CODES.UNKNOWN_ERROR
                // Gate hints ("Please close your mouth") are more useful than the generic copy.
                const hint = 'hint' in data ? (data.hint as string | undefined) : undefined
                setError(mapped, hint ?? null)
                setAnalysisStatus('error')
                toast({
                    title: 'Analysis Failed',
                    description: hint
                        ? hint
                        : mapped === ERROR_CODES.NO_FACE_DETECTED
                        ? 'No face detected. Please upload a clear front-facing photo.'
                        : mapped === ERROR_CODES.MULTIPLE_FACES_DETECTED
                        ? 'Multiple faces detected. Please upload an image with a single face.'
                        : mapped === ERROR_CODES.FILE_TOO_LARGE
                        ? 'File too large. Please upload an image under 5MB.'
                        : mapped === ERROR_CODES.NOT_A_REAL_FACE
                        ? 'This does not look like a photo of a person. Please upload a real photograph.'
                        : mapped === ERROR_CODES.EXPRESSION_NOT_NEUTRAL
                        ? 'Please use a neutral expression, with your face relaxed and mouth closed.'
                        : 'Could not analyze the image. Please try again.',
                    variant: 'destructive',
                })
                return
            }

            const okData = data as BackendAnalyzeResponse

            if (!isRegionRerun) {
                setProgressStep(PROCESSING_STEPS[3])
                await new Promise((resolve) => setTimeout(resolve, 200))
                setProgressStep(PROCESSING_STEPS[4])
                await new Promise((resolve) => setTimeout(resolve, 200))
            }

            const suggestions = Array.isArray(okData?.suggestions) ? okData.suggestions : []
            const analysisResult: AnalysisResult = {
                score: Math.round(Number(okData?.score ?? 0)),
                metrics: okData?.metrics || { symmetry: 0, proportions: 0, balance: 0 },
                landmarks: Array.isArray(okData?.landmarks) ? okData.landmarks : [],
                overlayTypeHints: okData?.overlayTypeHints || {
                    points: true,
                    outline: true,
                    mesh: false,
                },
                ratios: Array.isArray(okData?.ratios) ? okData.ratios : [],
                recommendations: Array.isArray(okData?.recommendations) ? okData.recommendations : [],
                recommendation_items: Array.isArray(okData?.recommendation_items) ? okData.recommendation_items : undefined,
                suggestions,
                region: okData?.region,
                gates: okData?.gates,
                notes: [
                    suggestions.length > 0
                        ? 'Analysis computed by backend models (beauty score, geometry diagnostics, and catalog suggestions).'
                        : 'Analysis computed by backend models (beauty score + feature model).',
                    'Images are processed temporarily for inference; not stored permanently by this app.',
                    'For best results, use a clear, front-facing photo with good lighting.',
                ],
            }

            setResult(analysisResult)
            setAnalysisStatus('success')

            // Save to Firestore. Skipped for an override re-run: changing the
            // comparison group re-reads the same photo, it is not a new analysis, and
            // saving each change would litter the profile with near-duplicates.
            if (user && !isRegionRerun) {
                try {
                    console.log('Saving analysis result for user:', user.uid)
                    const savedId = await saveAnalysisResult(user.uid, analysisResult)
                    console.log('Analysis saved successfully with ID:', savedId)
                    await loadAnalyses() // Refresh the list
                    toast({
                        title: 'Analysis Complete!',
                        description: `Your aesthetic score: ${analysisResult.score}/100. Results saved to your profile.`,
                    })
                } catch (saveError: any) {
                    console.error('Failed to save analysis:', saveError)
                    console.error('Error details:', {
                        code: saveError?.code,
                        message: saveError?.message,
                        stack: saveError?.stack
                    })
                    toast({
                        title: 'Analysis Complete!',
                        description: `Your aesthetic score: ${analysisResult.score}/100. Note: Failed to save to profile. ${saveError?.message || ''}`,
                        variant: 'destructive',
                    })
                }
            } else {
                console.warn('User not logged in, skipping save')
                toast({
                    title: 'Analysis Complete!',
                    description: `Your aesthetic score: ${analysisResult.score}/100. Sign in to save results.`,
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
            setRegionPending(false)
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
        setShowAnalysis(false)
        loadAnalyses()
    }

    const handleSignOut = async () => {
        try {
            await signOutUser()
            toast({
                title: "Signed Out",
                description: "You have been successfully signed out.",
            })
        } catch (error: any) {
            console.error("Sign out error:", error)
            toast({
                title: "Sign Out Failed",
                description: error.message || "An unexpected error occurred.",
                variant: "destructive",
            })
        }
    }

    // Show analysis UI if active
    if (showAnalysis) {
        if (error) {
            return (
                <div className="min-h-[calc(100vh-6rem)] bg-background flex items-center justify-center p-4">
                    <div className="w-full max-w-2xl">
                        <ErrorState errorCode={error} hint={errorHint} onRetry={handleRetry} onCancel={handleCancel} />
                    </div>
                </div>
            )
        }

        if (result && analysisStatus === 'success') {
            return (
                <div className="min-h-[calc(100vh-6rem)] bg-background flex items-center justify-center p-4">
                    <div className="w-full max-w-6xl">
                        <ResultsDashboard
                            result={result}
                            onAnalyzeAnother={handleAnalyzeAnother}
                            onDelete={handleDelete}
                            onRegionChange={(region) => processAnalysis(region)}
                            regionPending={regionPending}
                        />
                    </div>
                </div>
            )
        }

        if (isProcessing || analysisStatus === 'processing') {
            return (
                <div className="min-h-[calc(100vh-6rem)] bg-background flex items-center justify-center p-4">
                    <div className="w-full max-w-2xl">
                        <ProcessingStepper onCancel={handleCancel} />
                    </div>
                </div>
            )
        }

        if (previewUrl && selectedFile) {
            return (
                <div className="min-h-[calc(100vh-6rem)] bg-background py-8">
                    <div className="container mx-auto px-4 max-w-4xl">
                        <PreviewPanel
                            onAnalyze={handleAnalyze}
                            onRemove={handleRemove}
                            onChange={handleChange}
                            isAnalyzing={isProcessing}
                        />
                    </div>
                </div>
            )
        }

        // Show upload interface
        return (
            <div className="min-h-[calc(100vh-6rem)] relative overflow-hidden bg-background">
                <div className="fixed inset-0 -z-10">
                    <div className="absolute inset-0 bg-background"></div>
                    <div className="absolute top-1/4 left-1/4 w-[30rem] h-[30rem] bg-primary/10 rounded-full blur-[120px] mix-blend-screen animate-pulse-slow"></div>
                    <div className="absolute bottom-1/4 right-1/4 w-[30rem] h-[30rem] bg-secondary/10 rounded-full blur-[120px] mix-blend-screen"></div>
                    <div className="absolute inset-0 bg-grid-gold/[0.02]"></div>
                </div>

                <div className="container mx-auto px-4 py-16 max-w-5xl relative z-10">
                    <div className="mb-12 text-center">
                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-black/5 backdrop-blur-sm border border-black/5 text-primary text-sm font-medium mb-6">
                            <Sparkles className="h-4 w-4" />
                            Begin Your Consultation
                        </div>
                        <h1 className="text-5xl md:text-6xl font-bold mb-6 text-foreground">
                            <span className="text-gradient">Upload Your Photo</span>
                        </h1>
                        <p className="text-xl text-muted-foreground w-full max-w-2xl mx-auto leading-relaxed font-light">
                            Provide a clear, well-lit portrait to receive your personalized aesthetic and beauty analysis.
                        </p>
                    </div>

                    <UploadDropzone onFileSelect={handleFileSelect} disabled={isProcessing} />

                    <div className="grid md:grid-cols-3 gap-6 mt-12">
                        <div className="bg-black/5 border border-black/5 rounded-2xl p-6 glass-panel hover:bg-black/5 transition-colors">
                            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-yellow-500 to-yellow-600 flex items-center justify-center mb-4">
                                <ImageIcon className="h-6 w-6 text-foreground" />
                            </div>
                            <h3 className="font-semibold text-foreground mb-2">High Quality Photos</h3>
                            <p className="text-sm text-muted-foreground">Submit a bright, clear photo for the most accurate beauty suggestions.</p>
                        </div>
                        <div className="bg-black/5 border border-black/5 rounded-2xl p-6 glass-panel hover:bg-black/5 transition-colors">
                            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary to-amber-600 flex items-center justify-center mb-4">
                                <Sparkles className="h-6 w-6 text-foreground" />
                            </div>
                            <h3 className="font-semibold text-foreground mb-2">Professional Insights</h3>
                            <p className="text-sm text-muted-foreground">Our intelligent system analyzes your unique proportions.</p>
                        </div>
                        <div className="bg-black/5 border border-black/5 rounded-2xl p-6 glass-panel hover:bg-black/5 transition-colors">
                            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-yellow-500 to-yellow-600 flex items-center justify-center mb-4">
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

    // Show regular dashboard
    return (
        <div className="min-h-[calc(100vh-6rem)] bg-background text-foreground pt-10 pb-16 relative overflow-hidden">
            <div className="absolute inset-0 -z-10 overflow-hidden">
                <div className="absolute top-0 right-1/4 w-[30rem] h-[30rem] bg-primary/10 rounded-full blur-[150px] mix-blend-screen" />
                <div className="absolute bottom-1/4 left-1/4 w-[30rem] h-[30rem] bg-secondary/10 rounded-full blur-[150px] mix-blend-screen" />
                <div className="absolute inset-0 bg-grid-gold/[0.02]" />
            </div>

            <div className="container mx-auto px-4 max-w-6xl relative z-10">

                {/* Dashboard Header */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-12 border-b border-black/5 pb-8 gap-6">
                    <div className="flex items-center gap-6">
                        {user?.photoURL ? (
                            <img
                                src={user.photoURL}
                                alt={user.displayName || "User"}
                                className="h-20 w-20 rounded-full border-2 border-primary/30 object-cover shadow-[0_0_20px_rgba(255,215,0,0.2)]"
                            />
                        ) : (
                            <div className="h-20 w-20 rounded-full border border-white/20 bg-gradient-to-br from-primary to-accent flex items-center justify-center relative shadow-[0_0_20px_rgba(255,215,0,0.2)]">
                                <span className="text-3xl font-bold text-foreground">
                                    {user?.displayName?.[0] || user?.email?.[0]?.toUpperCase() || "U"}
                                </span>
                                <div className="absolute bottom-0 right-0 h-4 w-4 bg-yellow-500 rounded-full border-2 border-black" />
                            </div>
                        )}
                        <div>
                            <div className="flex items-center gap-2 mb-2">
                                <h1 className="text-3xl font-bold">
                                    {user?.displayName ? `${user.displayName}'s Profile` : 'Beauty Profile'}
                                </h1>
                                <ShieldCheck className="h-5 w-5 text-primary" />
                            </div>
                            <p className="text-muted-foreground flex items-center gap-2">
                                <Sparkles className="h-4 w-4" />
                                {user?.email || 'Member Identity Verified'}
                            </p>
                        </div>
                    </div>

                    <div className="flex gap-4 w-full md:w-auto">
                        <Button 
                            variant="outline" 
                            onClick={handleSignOut}
                            className="w-full md:w-auto bg-black/5 border-black/5 hover:bg-black/5 text-foreground"
                        >
                            <LogOut className="h-4 w-4 mr-2 text-muted-foreground" />
                            Sign Out
                        </Button>
                        <Button 
                            onClick={handleStartAnalysis}
                            className="w-full md:w-auto bg-gradient-to-r from-primary to-secondary hover:opacity-90 text-black font-semibold shadow-[0_0_20px_-5px_hsl(45,100%,50%,0.5)]"
                        >
                            <Sparkles className="h-4 w-4 mr-2" />
                            New Consultation
                        </Button>
                    </div>
                </div>

                {/* Dashboard Content */}
                <div className="grid lg:grid-cols-3 gap-8">

                    <div className="lg:col-span-2 space-y-8">
                        <div className="flex items-center justify-between">
                            <h2 className="text-2xl font-bold flex items-center gap-2">
                                <History className="h-5 w-5 text-primary" />
                                Consultation History
                            </h2>
                        </div>

                        {loading ? (
                            <div className="flex items-center justify-center py-12">
                                <Loader2 className="h-8 w-8 text-primary animate-spin" />
                            </div>
                        ) : analyses.length === 0 ? (
                            <div className="bg-black/5 border border-black/5 rounded-2xl p-12 text-center glass-panel">
                                <History className="h-12 w-12 text-muted-foreground mx-auto mb-4 opacity-50" />
                                <h3 className="text-lg font-semibold text-foreground mb-2">No Consultations Yet</h3>
                                <p className="text-muted-foreground mb-6">Start your first facial analysis to see your results here.</p>
                                <Button 
                                    onClick={handleStartAnalysis}
                                    className="bg-gradient-to-r from-primary to-secondary hover:opacity-90 text-black"
                                >
                                    <Sparkles className="mr-2 h-4 w-4" />
                                    Start Analysis
                                </Button>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {analyses.map((analysis, idx) => {
                                    const traits = getTopTraits(analysis.result.metrics)
                                    return (
                                        <motion.div
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: idx * 0.1 }}
                                            key={analysis.id || idx}
                                            className="bg-black/40 border border-black/5 rounded-2xl p-6 glass-panel hover:bg-black/5 transition-colors group"
                                        >
                                            <div className="flex items-center justify-between flex-wrap gap-4">
                                                <div className="flex items-center gap-4">
                                                    <div className="h-16 w-16 rounded-xl bg-gradient-to-br from-white/10 to-transparent border border-black/5 flex items-center justify-center">
                                                        <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
                                                            {analysis.result.score}
                                                        </span>
                                                    </div>
                                                    <div>
                                                        <div className="text-foreground font-semibold mb-1">
                                                            Aesthetic Score {analysis.result.score}/100
                                                        </div>
                                                        <div className="text-sm text-muted-foreground">
                                                            {formatDate(analysis.createdAt as Date)}
                                                        </div>
                                                    </div>
                                                </div>
                                                <div className="flex items-center gap-3 flex-wrap">
                                                    {traits.map((trait, tIdx) => (
                                                        <div key={tIdx} className="px-3 py-1 rounded-full bg-black/5 border border-black/5 text-xs text-primary">
                                                            {trait}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </motion.div>
                                    )
                                })}
                            </div>
                        )}
                    </div>

                    <div className="space-y-8">
                        <div className="bg-black/5 border border-black/5 rounded-3xl p-8 glass-panel">
                            <h3 className="text-xl font-bold mb-6 text-foreground">Beauty Metrics</h3>
                            {loading ? (
                                <div className="flex items-center justify-center py-8">
                                    <Loader2 className="h-6 w-6 text-primary animate-spin" />
                                </div>
                            ) : averageMetrics.totalAnalyses === 0 ? (
                                <div className="text-center py-8">
                                    <p className="text-muted-foreground text-sm">Complete your first analysis to see metrics</p>
                                </div>
                            ) : (
                                <div className="space-y-6">
                                    <div>
                                        <div className="flex justify-between text-sm mb-2">
                                            <span className="text-muted-foreground">Average Symmetry</span>
                                            <span className="text-foreground font-semibold">{averageMetrics.averageSymmetry}%</span>
                                        </div>
                                        <div className="h-2 w-full bg-black/50 rounded-full overflow-hidden">
                                            <div 
                                                className="h-full bg-gradient-to-r from-yellow-400 to-amber-500 transition-all duration-500"
                                                style={{ width: `${averageMetrics.averageSymmetry}%` }}
                                            />
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-2">
                                            <span className="text-muted-foreground">Average Proportions</span>
                                            <span className="text-foreground font-semibold">{averageMetrics.averageProportions}%</span>
                                        </div>
                                        <div className="h-2 w-full bg-black/50 rounded-full overflow-hidden">
                                            <div 
                                                className="h-full bg-gradient-to-r from-amber-400 to-yellow-500 transition-all duration-500"
                                                style={{ width: `${averageMetrics.averageProportions}%` }}
                                            />
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-2">
                                            <span className="text-muted-foreground">Average Balance</span>
                                            <span className="text-foreground font-semibold">{averageMetrics.averageBalance}%</span>
                                        </div>
                                        <div className="h-2 w-full bg-black/50 rounded-full overflow-hidden">
                                            <div 
                                                className="h-full bg-gradient-to-r from-yellow-400 to-amber-500 transition-all duration-500"
                                                style={{ width: `${averageMetrics.averageBalance}%` }}
                                            />
                                        </div>
                                    </div>
                                    <div className="pt-4 border-t border-black/5">
                                        <div className="flex justify-between text-sm mb-2">
                                            <span className="text-muted-foreground">Consultations Completed</span>
                                            <span className="text-foreground font-semibold">{averageMetrics.totalAnalyses} {averageMetrics.totalAnalyses === 1 ? 'Profile' : 'Profiles'}</span>
                                        </div>
                                        <div className="h-2 w-full bg-black/50 rounded-full overflow-hidden">
                                            <div 
                                                className="h-full bg-gradient-to-r from-amber-400 to-yellow-500 transition-all duration-500"
                                                style={{ width: `${Math.min((averageMetrics.totalAnalyses / 10) * 100, 100)}%` }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                </div>

            </div>
        </div>
    )
}

export default function DashboardPage() {
    return (
        <ProtectedRoute>
            <DashboardContent />
        </ProtectedRoute>
    )
}
