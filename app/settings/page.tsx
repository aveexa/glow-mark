"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Sparkles, Settings as SettingsIcon, ChevronLeft, Loader2 } from "lucide-react"
import Link from "next/link"
import { ProtectedRoute } from "@/components/protected-route"
import { useToast } from "@/hooks/use-toast"
import { cn } from "@/lib/utils"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5001"

function SettingsContent() {
    const { toast } = useToast()
    const [useLLM, setUseLLM] = useState(false)
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)

    useEffect(() => {
        let active = true
        ;(async () => {
            try {
                const res = await fetch(`${BACKEND_URL}/api/settings/summary`)
                const data = await res.json()
                if (active) setUseLLM(Boolean(data?.use_llm))
            } catch {
                // Environment/firestore issues aside, fall back to the persisted default.
            } finally {
                if (active) setLoading(false)
            }
        })()
        return () => {
            active = false
        }
    }, [])

    const toggleSummary = async () => {
        const next = !useLLM
        setSaving(true)
        try {
            const res = await fetch(`${BACKEND_URL}/api/settings/summary`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ use_llm: next }),
            })
            const data = await res.json()
            setUseLLM(Boolean(data?.use_llm))
            toast({
                title: next ? "AI Summary enabled" : "AI Summary disabled",
                description: next
                    ? "Your analyses will include an AI-polished summary."
                    : "Your analyses will use the standard summary.",
            })
        } catch {
            toast({
                title: "Failed to save setting",
                description: "Please try again.",
                variant: "destructive",
            })
        } finally {
            setSaving(false)
        }
    }

    return (
        <div className="min-h-[calc(100vh-6rem)] bg-background text-foreground pt-10 pb-16 relative overflow-hidden">
            <div className="absolute inset-0 -z-10 overflow-hidden">
                <div className="absolute top-0 right-1/4 w-[30rem] h-[30rem] bg-primary/10 rounded-full blur-[150px] mix-blend-screen" />
                <div className="absolute bottom-1/4 left-1/4 w-[30rem] h-[30rem] bg-secondary/10 rounded-full blur-[150px] mix-blend-screen" />
                <div className="absolute inset-0 bg-grid-gold/[0.02]" />
            </div>

            <div className="container mx-auto px-4 max-w-3xl relative z-10">
                <Link
                    href="/dashboard"
                    className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6"
                >
                    <ChevronLeft className="h-4 w-4" />
                    Back to Dashboard
                </Link>

                <div className="flex items-center gap-4 mb-8 border-b border-black/5 pb-6">
                    <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-[0_0_20px_rgba(255,215,0,0.3)]">
                        <SettingsIcon className="h-6 w-6 text-foreground" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold">Settings</h1>
                        <p className="text-sm text-muted-foreground">Manage your GlowMark preferences.</p>
                    </div>
                </div>

                <div className="bg-black/5 border border-black/5 rounded-2xl p-6 glass-panel">
                    <div className="flex items-start justify-between gap-6">
                        <div className="flex items-start gap-4">
                            <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-primary to-amber-600 flex items-center justify-center shrink-0">
                                <Sparkles className="h-5 w-5 text-foreground" />
                            </div>
                            <div>
                                <h2 className="font-semibold text-foreground">AI Summary</h2>
                                <p className="text-sm text-muted-foreground mt-1 max-w-md">
                                    Generate a polished, natural-language summary of your analysis
                                    recommendations alongside the standard insights.
                                </p>
                            </div>
                        </div>

                        {loading ? (
                            <div className="h-8 w-16 bg-black/5 rounded-full animate-pulse shrink-0" />
                        ) : (
                            <button
                                onClick={toggleSummary}
                                disabled={saving}
                                aria-label="Toggle AI summary"
                                aria-pressed={useLLM}
                                className="relative flex items-center rounded-full h-8 w-16 shrink-0 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                                style={{ backgroundColor: useLLM ? "hsl(45 100% 50%)" : "rgb(0 0 0 / 0.12)" }}
                            >
                                <motion.span
                                    layout
                                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                                    className={cn(
                                        "block h-6 w-6 rounded-full bg-white shadow-md",
                                        useLLM ? "ml-auto mr-1" : "ml-1"
                                    )}
                                />
                            </button>
                        )}
                    </div>

                    {saving && (
                        <div className="flex items-center gap-2 mt-4 text-sm text-muted-foreground">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Saving...
                        </div>
                    )}

                    <div className="mt-5 pt-4 border-t border-black/5 text-xs text-muted-foreground">
                        Toggling this on or off takes effect immediately. Past analyses keep their stored summaries.
                    </div>
                </div>
            </div>
        </div>
    )
}

export default function SettingsPage() {
    return (
        <ProtectedRoute>
            <SettingsContent />
        </ProtectedRoute>
    )
}
