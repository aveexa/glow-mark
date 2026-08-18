"use client"

import { useState, Suspense, FormEvent } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Brain, Mail, Lock, Sparkles, MoveRight, Github, Loader2, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import Link from "next/link"
import { useSearchParams, useRouter } from "next/navigation"
import { signUpWithEmail, signInWithEmail, signInWithGoogle, resetPassword } from "@/lib/firebase/auth"
import { auth } from "@/lib/firebase/config"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/contexts/auth-context"

export default function LoginPage() {
    return (
        <Suspense fallback={
            <div className="min-h-screen flex items-center justify-center bg-background text-foreground">
                <div className="flex flex-col items-center gap-4">
                    <Brain className="h-8 w-8 text-primary animate-pulse" />
                    <p className="text-muted-foreground animate-pulse">Initializing Biometric Hub...</p>
                </div>
            </div>
        }>
            <LoginContent />
        </Suspense>
    )
}

function LoginContent() {
    const searchParams = useSearchParams()
    const router = useRouter()
    const { toast } = useToast()
    const { user } = useAuth()
    const initialTab = searchParams.get("tab") === "register" ? "register" : "login"
    const [activeTab, setActiveTab] = useState<"login" | "register">(initialTab)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [email, setEmail] = useState("")
    const [password, setPassword] = useState("")
    const [displayName, setDisplayName] = useState("")
    const [forgotPassword, setForgotPassword] = useState(false)
    const [resetEmail, setResetEmail] = useState("")

    // Redirect if already logged in
    if (user) {
        router.push("/dashboard")
        return null
    }

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault()
        setError(null)
        setLoading(true)

        try {
            if (activeTab === "register") {
                await signUpWithEmail(email, password, displayName)
                toast({
                    title: "Account created!",
                    description: "Welcome to GlowMark. Redirecting to dashboard...",
                })
                router.push("/dashboard")
            } else {
                await signInWithEmail(email, password)
                toast({
                    title: "Welcome back!",
                    description: "Successfully signed in.",
                })
                router.push("/dashboard")
            }
        } catch (error: any) {
            console.error("Auth error:", error)
            let errorMessage = "An error occurred. Please try again."
            
            switch (error.code) {
                case "auth/email-already-in-use":
                    errorMessage = "This email is already registered. Please sign in instead."
                    break
                case "auth/invalid-email":
                    errorMessage = "Invalid email address."
                    break
                case "auth/weak-password":
                    errorMessage = "Password should be at least 6 characters."
                    break
                case "auth/user-not-found":
                    errorMessage = "No account found with this email."
                    break
                case "auth/wrong-password":
                    errorMessage = "Incorrect password."
                    break
                case "auth/too-many-requests":
                    errorMessage = "Too many failed attempts. Please try again later."
                    break
                default:
                    errorMessage = error.message || errorMessage
            }
            
            setError(errorMessage)
            toast({
                title: "Authentication failed",
                description: errorMessage,
                variant: "destructive",
            })
        } finally {
            setLoading(false)
        }
    }

    const handleGoogleSignIn = async () => {
        setError(null)
        setLoading(true)

        try {
            await signInWithGoogle()
            toast({
                title: "Success!",
                description: "Signed in with Google.",
            })
            router.push("/dashboard")
        } catch (error: any) {
            console.error("Google sign-in error:", error)

            // If Auth already established a session, treat as success (safety net).
            if (auth.currentUser) {
                toast({
                    title: "Success!",
                    description: "Signed in with Google.",
                })
                router.push("/dashboard")
                return
            }

            // Only surface real Firebase Auth failures as Google sign-in errors.
            if (typeof error?.code === "string" && error.code.startsWith("auth/")) {
                let errorMessage = "Failed to sign in with Google. Please try again."

                if (error.code === "auth/popup-closed-by-user") {
                    errorMessage = "Sign-in popup was closed."
                } else if (error.code === "auth/popup-blocked") {
                    errorMessage = "Popup was blocked. Please allow popups for this site."
                }

                setError(errorMessage)
                toast({
                    title: "Google sign-in failed",
                    description: errorMessage,
                    variant: "destructive",
                })
            }
        } finally {
            setLoading(false)
        }
    }

    const handlePasswordReset = async () => {
        if (!resetEmail) {
            setError("Please enter your email address.")
            return
        }

        setError(null)
        setLoading(true)

        try {
            await resetPassword(resetEmail)
            toast({
                title: "Password reset email sent!",
                description: "Check your inbox for reset instructions.",
            })
            setForgotPassword(false)
            setResetEmail("")
        } catch (error: any) {
            console.error("Password reset error:", error)
            let errorMessage = "Failed to send reset email. Please try again."
            
            if (error.code === "auth/user-not-found") {
                errorMessage = "No account found with this email."
            } else if (error.code === "auth/invalid-email") {
                errorMessage = "Invalid email address."
            }
            
            setError(errorMessage)
            toast({
                title: "Reset failed",
                description: errorMessage,
                variant: "destructive",
            })
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="relative min-h-[calc(100vh-6rem)] flex items-center justify-center overflow-hidden">
            {/* Immersive Background */}
            <div className="absolute inset-0 -z-10 bg-background overflow-hidden">
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-[120px] animate-pulse-slow mix-blend-screen" />
                <div className="absolute top-1/3 right-1/4 w-[28rem] h-[28rem] bg-secondary/20 rounded-full blur-[120px] animate-pulse-slow mix-blend-screen" style={{ animationDelay: "1s" }} />
                <div className="absolute bottom-1/4 left-1/2 w-[32rem] h-[32rem] bg-accent/20 rounded-full blur-[120px] animate-pulse-slow mix-blend-screen" style={{ animationDelay: "2s" }} />
                <div className="absolute inset-0 bg-grid-gold/[0.02]" />
            </div>

            <div className="container mx-auto px-4 py-8 relative z-10">
                <div className="max-w-6xl mx-auto flex flex-col lg:flex-row shadow-[0_0_50px_-15px_rgba(255,215,0,0.3)] rounded-[2rem] overflow-hidden border border-black/5 glass-panel">

                    {/* Left Side - Visual Aesthetic */}
                    <div className="lg:w-1/2 relative hidden lg:flex flex-col justify-between p-12 bg-black/40 border-r border-black/5 overflow-hidden">
                        {/* Animated Geometry Background */}
                        <div className="absolute inset-0 -z-10 bg-[url('https://images.unsplash.com/photo-1634152962476-4b8a00e1915c?q=80&w=1200&auto=format&fit=crop')] bg-cover bg-center opacity-30 mix-blend-luminosity grayscale contrast-150" />
                        <div className="absolute inset-0 -z-10 bg-gradient-to-t from-[#030014] via-[#030014]/80 to-transparent" />
                        <div className="absolute inset-0 -z-10 bg-gradient-to-r from-primary/20 via-transparent to-transparent opacity-50" />

                        <div className="relative z-10 flex items-center gap-2">
                            <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary via-amber-500 to-accent text-foreground shadow-[0_0_20px_rgba(255,215,0,0.4)]">
                                <Brain className="h-5 w-5" />
                            </div>
                            <span className="text-2xl font-bold tracking-tight text-foreground">
                                Glow<span className="text-primary">Mark</span>
                            </span>
                        </div>

                        <div className="relative z-10 mt-20">
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.2 }}
                            >
                                <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-black/5 border border-black/5 text-primary text-sm font-medium mb-6 backdrop-blur-md">
                                    <Sparkles className="h-4 w-4" />
                                    Global Identity
                                </div>
                                <h1 className="text-4xl xl:text-5xl font-extrabold text-foreground leading-tight mb-6">
                                    Access Your <span className="text-gradient">Biometric Hub</span>
                                </h1>
                                <p className="text-lg text-muted-foreground font-light max-w-md leading-relaxed">
                                    Enter the portal to manage your scans, track aesthetic metrics over time, and configure personalized neural models.
                                </p>
                            </motion.div>
                        </div>

                        {/* Decorative Grid */}
                        <div className="relative z-10 grid grid-cols-2 gap-4 mt-16 opacity-60">
                            {[...Array(4)].map((_, i) => (
                                <div key={i} className="h-1 bg-black/5 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full bg-primary"
                                        initial={{ width: "0%" }}
                                        animate={{ width: `${Math.random() * 60 + 20}%` }}
                                        transition={{ duration: 2, repeat: Infinity, repeatType: "reverse", delay: i * 0.2 }}
                                    />
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Right Side - Auth Form */}
                    <div className="w-full lg:w-1/2 p-8 md:p-12 lg:p-16 relative bg-background/80 backdrop-blur-md flex flex-col justify-center">

                        <div className="max-w-md w-full mx-auto">
                            {/* Tab Selector */}
                            <div className="flex p-1 bg-black/5 rounded-xl border border-black/5 mb-8 relative">
                                {["login", "register"].map((tab) => (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab as any)}
                                        className={`flex-1 py-2.5 text-sm font-semibold rounded-lg z-10 transition-colors uppercase tracking-wider ${activeTab === tab ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                                            }`}
                                    >
                                        {tab === "login" ? "Sign In" : "Create Account"}
                                    </button>
                                ))}

                                {/* Active Tab Background Indicator */}
                                <motion.div
                                    className="absolute top-1 bottom-1 w-[calc(50%-4px)] bg-primary/20 backdrop-blur-md border border-primary/30 rounded-lg shadow-[0_0_15px_rgba(255,215,0,0.3)]"
                                    initial={false}
                                    animate={{ left: activeTab === "login" ? "4px" : "calc(50%)" }}
                                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                                />
                            </div>

                            <div className="mb-8">
                                <h2 className="text-2xl font-bold text-foreground mb-2">
                                    {activeTab === "login" ? "Welcome back" : "Initialize Identity"}
                                </h2>
                                <p className="text-muted-foreground">
                                    {activeTab === "login"
                                        ? "Enter your credentials to access your dashboard."
                                        : "Establish your secure biometric profile."}
                                </p>
                            </div>

                            {/* Form Content */}
                            <AnimatePresence mode="wait">
                                <motion.div
                                    key={activeTab}
                                    initial={{ opacity: 0, x: 10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -10 }}
                                    transition={{ duration: 0.2 }}
                                >
                                    {forgotPassword ? (
                                        <div className="space-y-5">
                                            <div className="space-y-2">
                                                <Label htmlFor="reset-email" className="text-muted-foreground ml-1">Email Address</Label>
                                                <div className="relative">
                                                    <Mail className="absolute left-3 top-3 h-5 w-5 text-muted-foreground/50" />
                                                    <Input
                                                        id="reset-email"
                                                        type="email"
                                                        placeholder="your@email.com"
                                                        value={resetEmail}
                                                        onChange={(e) => setResetEmail(e.target.value)}
                                                        className="bg-black/50 border-black/5 h-12 pl-10 focus-visible:ring-primary focus-visible:border-primary text-foreground"
                                                    />
                                                </div>
                                            </div>
                                            <div className="flex gap-3">
                                                <Button
                                                    type="button"
                                                    variant="outline"
                                                    onClick={() => setForgotPassword(false)}
                                                    className="flex-1 glass-panel border-black/5"
                                                >
                                                    Cancel
                                                </Button>
                                                <Button
                                                    type="button"
                                                    onClick={handlePasswordReset}
                                                    disabled={loading || !resetEmail}
                                                    className="flex-1 bg-gradient-to-r from-primary to-accent hover:opacity-90"
                                                >
                                                    {loading ? (
                                                        <Loader2 className="h-4 w-4 animate-spin" />
                                                    ) : (
                                                        "Send Reset Link"
                                                    )}
                                                </Button>
                                            </div>
                                        </div>
                                    ) : (
                                        <form className="space-y-5" onSubmit={handleSubmit}>

                                            {activeTab === "register" && (
                                                <div className="space-y-2">
                                                    <Label htmlFor="name" className="text-muted-foreground ml-1">Full Name</Label>
                                                    <div className="relative">
                                                        <Brain className="absolute left-3 top-3 h-5 w-5 text-muted-foreground/50" />
                                                        <Input
                                                            id="name"
                                                            placeholder="John Doe"
                                                            value={displayName}
                                                            onChange={(e) => setDisplayName(e.target.value)}
                                                            required={activeTab === "register"}
                                                            className="bg-black/50 border-black/5 h-12 pl-10 focus-visible:ring-primary focus-visible:border-primary text-foreground"
                                                        />
                                                    </div>
                                                </div>
                                            )}

                                            <div className="space-y-2">
                                                <Label htmlFor="email" className="text-muted-foreground ml-1">Email Address</Label>
                                                <div className="relative">
                                                    <Mail className="absolute left-3 top-3 h-5 w-5 text-muted-foreground/50" />
                                                    <Input
                                                        id="email"
                                                        type="email"
                                                        placeholder="your@email.com"
                                                        value={email}
                                                        onChange={(e) => setEmail(e.target.value)}
                                                        required
                                                        className="bg-black/50 border-black/5 h-12 pl-10 focus-visible:ring-primary focus-visible:border-primary text-foreground"
                                                    />
                                                </div>
                                            </div>

                                            <div className="space-y-2">
                                                <div className="flex justify-between">
                                                    <Label htmlFor="password" className="text-muted-foreground ml-1">Password</Label>
                                                    {activeTab === "login" && (
                                                        <button
                                                            type="button"
                                                            onClick={() => setForgotPassword(true)}
                                                            className="text-xs text-primary hover:text-accent font-medium"
                                                        >
                                                            Forgot password?
                                                        </button>
                                                    )}
                                                </div>
                                                <div className="relative">
                                                    <Lock className="absolute left-3 top-3 h-5 w-5 text-muted-foreground/50" />
                                                    <Input
                                                        id="password"
                                                        type="password"
                                                        placeholder="••••••••"
                                                        value={password}
                                                        onChange={(e) => setPassword(e.target.value)}
                                                        required
                                                        minLength={6}
                                                        className="bg-black/50 border-black/5 h-12 pl-10 focus-visible:ring-primary focus-visible:border-primary text-foreground"
                                                    />
                                                </div>
                                            </div>

                                            {error && (
                                                <div className="flex items-center gap-2 p-3 bg-destructive/10 border border-destructive/30 rounded-lg text-sm text-destructive">
                                                    <AlertCircle className="h-4 w-4 flex-shrink-0" />
                                                    <span>{error}</span>
                                                </div>
                                            )}

                                            <Button
                                                type="submit"
                                                disabled={loading}
                                                className="w-full h-12 mt-6 bg-gradient-to-r from-primary to-accent hover:opacity-90 shadow-[0_0_20px_-5px_rgba(255,215,0,0.5)] border border-black/5 text-lg font-semibold rounded-xl group relative overflow-hidden disabled:opacity-50"
                                            >
                                                <span className="relative z-10 flex items-center justify-center">
                                                    {loading ? (
                                                        <>
                                                            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                                                            {activeTab === "login" ? "Signing in..." : "Creating account..."}
                                                        </>
                                                    ) : (
                                                        <>
                                                            {activeTab === "login" ? "Sign In" : "Create Account"}
                                                            <MoveRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                                                        </>
                                                    )}
                                                </span>
                                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
                                            </Button>
                                        </form>
                                    )}

                                    {/* Social Login Separator */}
                                    <div className="mt-8 relative flex items-center py-5">
                                        <div className="flex-grow border-t border-black/5"></div>
                                        <span className="flex-shrink-0 mx-4 text-xs font-medium text-muted-foreground uppercase tracking-widest">or authenticate via</span>
                                        <div className="flex-grow border-t border-black/5"></div>
                                    </div>

                                    {!forgotPassword && (
                                        <>
                                            {/* Social Login Separator */}
                                            <div className="mt-8 relative flex items-center py-5">
                                                <div className="flex-grow border-t border-black/5"></div>
                                                <span className="flex-shrink-0 mx-4 text-xs font-medium text-muted-foreground uppercase tracking-widest">or continue with</span>
                                                <div className="flex-grow border-t border-black/5"></div>
                                            </div>

                                            {/* Social Buttons */}
                                            <Button
                                                type="button"
                                                variant="outline"
                                                onClick={handleGoogleSignIn}
                                                disabled={loading}
                                                className="w-full glass-panel border-black/5 hover:bg-black/5 h-11 text-muted-foreground disabled:opacity-50"
                                            >
                                                {loading ? (
                                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                ) : (
                                                    <svg viewBox="0 0 24 24" className="mr-2 h-4 w-4" aria-hidden="true" focusable="false" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                                                        <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"></path>
                                                        <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"></path>
                                                        <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"></path>
                                                        <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"></path>
                                                    </svg>
                                                )}
                                                Continue with Google
                                            </Button>
                                        </>
                                    )}
                                </motion.div>
                            </AnimatePresence>

                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
