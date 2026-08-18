'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Sparkles, Shield, Zap, Brain, TrendingUp, Lock, ArrowRight } from 'lucide-react'
import { motion } from 'framer-motion'

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.2 }
  }
}

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: 'spring' as const, stiffness: 100, damping: 10 }
  }
}

export default function Home() {
  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      {/* Immersive Background */}
      <div className="fixed inset-0 -z-10 bg-grid-gold/[0.02]" />
      <div className="fixed inset-0 -z-10 flex justify-center items-center">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-[120px] animate-pulse-slow mix-blend-screen" />
        <div className="absolute top-1/3 right-1/4 w-[28rem] h-[28rem] bg-secondary/20 rounded-full blur-[120px] animate-pulse-slow mix-blend-screen" style={{ animationDelay: '1s' }} />
        <div className="absolute bottom-1/4 left-1/2 w-[32rem] h-[32rem] bg-accent/20 rounded-full blur-[120px] animate-pulse-slow mix-blend-screen" style={{ animationDelay: '2s' }} />
      </div>

      {/* Hero Section */}
      <div className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
        <div className="container mx-auto px-4 relative z-10">
          <motion.div
            className="text-center max-w-5xl mx-auto"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            <motion.div variants={itemVariants} className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full border border-primary/30 bg-primary/10 text-primary text-sm font-semibold mb-8 backdrop-blur-md shadow-[0_0_20px_rgba(255,215,0,0.3)]">
              <Sparkles className="h-4 w-4" />
              Advanced Aesthetics Studio
            </motion.div>

            <motion.h1 variants={itemVariants} className="text-6xl md:text-8xl font-extrabold mb-8 tracking-tight text-foreground leading-tight">
              Discover Your True <br />
              <span className="text-gradient">
                Radiance
              </span>
            </motion.h1>

            <motion.p variants={itemVariants} className="text-lg md:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto font-light leading-relaxed">
              Experience the future of beauty. Receive a personalized aesthetic analysis, discover perfect proportions, and get tailored skin and feature suggestions.
            </motion.p>

            <motion.div variants={itemVariants} className="flex flex-col sm:flex-row gap-6 justify-center items-center">
              <Link href="/dashboard">
                <Button size="lg" className="group text-lg px-8 py-7 h-auto bg-gradient-to-r from-primary to-secondary hover:opacity-90 transition-all duration-300 shadow-[0_0_40px_-5px_hsl(45,100%,50%,0.5)] rounded-2xl border border-black/5 text-black font-medium">
                  <span className="mr-2">Start Analysis</span>
                  <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
              <Link href="/privacy">
                <Button size="lg" variant="outline" className="text-lg px-8 py-7 h-auto glass-panel hover:bg-black/5 transition-all duration-300 rounded-2xl text-foreground font-medium">
                  <Lock className="mr-2 h-5 w-5 text-muted-foreground" />
                  Our Privacy Pledge
                </Button>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* Features Section */}
      <div className="container mx-auto px-4 py-24 relative z-10">
        <motion.div
          className="text-center mb-20"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6 tracking-tight">
            Engineered for <span className="text-gradient">Excellence</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto font-light">
            Leveraging mathematical facial modeling and neural networks to map the landscape of human aesthetics.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8 mb-24">
          {[
            {
              title: "Detailed Feature Analysis",
              desc: "Map your unique facial contours, eyes, and jawline to discover your perfect proportions.",
              icon: <Sparkles className="h-6 w-6 text-foreground" />,
              gradient: "from-yellow-600 via-amber-600 to-primary",
              image: "https://images.unsplash.com/photo-1542360215099-276f75605d82?q=80&w=600&auto=format&fit=crop"
            },
            {
              title: "100% Private & Secure",
              desc: "Rest easy knowing your photos are processed instantly and never stored on our servers.",
              icon: <Shield className="h-6 w-6 text-foreground" />,
              gradient: "from-primary via-yellow-600 to-secondary",
              image: "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?q=80&w=600&auto=format&fit=crop"
            },
            {
              title: "Instant Beauty Insights",
              desc: "Get tailored skin and aesthetic suggestions in under 5 seconds, straight from our experts.",
              icon: <Zap className="h-6 w-6 text-foreground" />,
              gradient: "from-secondary via-yellow-600 to-accent",
              image: "https://images.unsplash.com/photo-1551836022-d5d88e9218df?q=80&w=600&auto=format&fit=crop"
            }
          ].map((feature, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: idx * 0.15 }}
              className="h-full"
            >
              <Card className="glass-panel border-black/5 hover:border-primary/30 transition-all duration-500 overflow-hidden relative group h-full flex flex-col pt-0">
                <div className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-500`} />

                {/* Feature Image Banner */}
                <div className="relative h-48 w-full overflow-hidden shrink-0">
                  <img src={feature.image} alt={feature.title} className="w-full h-full object-cover opacity-80 group-hover:opacity-100 group-hover:scale-110 transition-all duration-700" />
                  <div className="absolute inset-0 bg-gradient-to-t from-[rgba(20,20,30,0.8)] to-transparent" />

                  {/* Floating Icon Base */}
                  <div className={`absolute -bottom-6 left-6 h-12 w-12 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center shadow-lg transform group-hover:-translate-y-2 group-hover:scale-110 transition-transform duration-500 shadow-[0_0_15px_rgba(255,255,255,0.2)] z-20`}>
                    {feature.icon}
                  </div>
                </div>

                <CardHeader className="pt-10 pb-6 relative z-10 flex-grow">
                  <CardTitle className="text-2xl mb-3 font-bold text-foreground group-hover:text-primary transition-colors">{feature.title}</CardTitle>
                  <CardDescription className="text-base text-muted-foreground leading-relaxed">
                    {feature.desc}
                  </CardDescription>
                </CardHeader>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Stats Section */}
        <motion.div
          className="relative neon-border rounded-3xl"
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
        >
          <div className="glass-panel rounded-3xl p-12 relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-secondary/5 to-accent/5 backdrop-blur-3xl" />
            <div className="relative z-10 grid grid-cols-2 md:grid-cols-4 gap-12 text-center">
              {[
                { val: "100%", label: "Privacy Enforced" },
                { val: "<5s", label: "Fast Results" },
                { val: "AI", label: "Smart Analysis" },
                { val: "68+", label: "Beauty Metrics" }
              ].map((stat, idx) => (
                <div key={idx}>
                  <div className="text-5xl font-extrabold text-gradient mb-3 drop-shadow-sm">
                    {stat.val}
                  </div>
                  <div className="text-sm font-medium tracking-wide text-muted-foreground uppercase">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Footer */}
      <footer className="border-t border-black/5 glass-panel mt-20 relative z-10">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center text-sm text-muted-foreground">
            <p className="mb-4 md:mb-0">
              © 2026 GlowMark Systems. Research & Educational Platform.
            </p>
            <div className="flex gap-6">
              <Link href="/privacy" className="hover:text-primary transition-colors font-medium">
                Privacy Policy
              </Link>
              <Link href="/dashboard" className="hover:text-primary transition-colors font-medium">
                Start Analysis
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
