import { Brain, Zap, UserCheck, Code } from "lucide-react"

export default function AboutPage() {
    return (
        <div className="min-h-[calc(100vh-6rem)] bg-background text-foreground pt-20 pb-16 relative overflow-hidden">
            <div className="absolute inset-0 -z-10 bg-[url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop')] bg-cover bg-center opacity-10 mix-blend-luminosity grayscale contrast-150" />
            <div className="absolute inset-0 -z-10 bg-gradient-to-t from-background via-background/80 to-transparent" />

            <div className="container mx-auto px-4 max-w-5xl relative z-10">
                <div className="text-center mb-20 animate-fade-in">
                    <h1 className="text-5xl md:text-7xl font-bold mb-6">
                        The Art of <span className="text-gradient">Aesthetics</span>
                    </h1>
                    <p className="text-xl text-muted-foreground max-w-3xl mx-auto font-light leading-relaxed">
                        GlowMark combines professional beauty expertise with advanced facial analysis to help you discover your perfect proportions and personalized aesthetic profile.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 gap-12 mt-12">
                    {/* Architecture */}
                    <div className="bg-black/5 border border-black/5 rounded-3xl p-8 glass-panel">
                        <div className="h-14 w-14 rounded-2xl bg-gradient-to-br from-primary to-amber-600 flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(255,215,0,0.4)]">
                            <Brain className="h-7 w-7 text-foreground" />
                        </div>
                        <h2 className="text-2xl font-bold mb-4">Expert Analysis</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            Our platform utilizes a highly secure, private assessment pipeline. When you upload your photo, it is evaluated instantly to provide tailored beauty insights, highlight your best features, and suggest personalized enhancements.
                        </p>
                    </div>

                    <div className="bg-black/5 border border-black/5 rounded-3xl p-8 glass-panel">
                        <div className="h-14 w-14 rounded-2xl bg-gradient-to-br from-secondary to-yellow-600 flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(255,215,0,0.4)]">
                            <Zap className="h-7 w-7 text-foreground" />
                        </div>
                        <h2 className="text-2xl font-bold mb-4">Instant Beauty Profile</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            Leveraging state-of-the-art beauty algorithms, your consultation is completed in seconds. You receive immediate, actionable feedback on facial symmetry, harmony, and personalized recommendations for styling and aesthetics.
                        </p>
                    </div>
                </div>

            </div>
        </div>
    )
}
