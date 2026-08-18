"use client"

import { motion } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import { 
  Brain, 
  Camera, 
  Scan, 
  Calculator, 
  TrendingUp, 
  Sparkles,
  ArrowRight,
  CheckCircle2,
  Zap,
  Shield
} from 'lucide-react'

const steps = [
  {
    icon: Camera,
    title: 'Upload Your Photo',
    description: 'Upload a clear, front-facing photo. We support JPEG, PNG, and WebP formats up to 5MB.',
    color: 'from-yellow-500 to-amber-500'
  },
  {
    icon: Scan,
    title: 'Face Detection',
    description: 'Our MediaPipe AI detects and locates your face in the image, ensuring optimal analysis conditions.',
    color: 'from-amber-500 to-orange-500'
  },
  {
    icon: Brain,
    title: 'Landmark Mapping',
    description: 'We map 468 facial landmarks including eyes, nose, mouth, jawline, and facial contours.',
    color: 'from-orange-500 to-yellow-500'
  },
  {
    icon: Calculator,
    title: 'Beauty Calculations',
    description: 'Advanced algorithms calculate symmetry, proportions, and balance using golden ratio principles.',
    color: 'from-yellow-500 to-amber-500'
  },
  {
    icon: TrendingUp,
    title: 'Score Generation',
    description: 'Your personalized aesthetic score (0-100) is calculated based on comprehensive facial analysis.',
    color: 'from-amber-500 to-yellow-500'
  },
  {
    icon: Sparkles,
    title: 'Get Recommendations',
    description: 'Receive personalized beauty suggestions and insights tailored to your unique facial features.',
    color: 'from-yellow-500 to-amber-500'
  }
]

const features = [
  {
    title: '468 Facial Landmarks',
    description: 'MediaPipe technology detects 468 precise facial points for comprehensive analysis'
  },
  {
    title: 'Golden Ratio Analysis',
    description: 'Compare your facial proportions against the mathematical golden ratio (1.618)'
  },
  {
    title: 'Symmetry Measurement',
    description: 'Calculate facial symmetry by analyzing eye, nose, and mouth alignment'
  },
  {
    title: 'Proportional Balance',
    description: 'Evaluate facial thirds, eye spacing, and feature relationships'
  },
  {
    title: 'Real-time Processing',
    description: 'Get results in seconds with client-side processing - no server delays'
  },
  {
    title: 'Privacy First',
    description: 'All processing happens in your browser. Images are never stored permanently'
  }
]

export default function HowItWorksPage() {
  return (
    <div className="min-h-[calc(100vh-6rem)] bg-background relative overflow-hidden">
      {/* Background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-background"></div>
        <div className="absolute top-1/4 left-1/4 w-[30rem] h-[30rem] bg-primary/10 rounded-full blur-[120px] mix-blend-screen animate-pulse-slow"></div>
        <div className="absolute bottom-1/4 right-1/4 w-[30rem] h-[30rem] bg-secondary/10 rounded-full blur-[120px] mix-blend-screen"></div>
        <div className="absolute inset-0 bg-grid-gold/[0.02]"></div>
      </div>

      <div className="container mx-auto px-4 py-16 max-w-6xl relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-black/5 border border-black/5 text-primary text-sm font-medium mb-6">
            <Brain className="h-4 w-4" />
            How We Analyze
          </div>
          <h1 className="text-5xl md:text-6xl font-bold mb-6 text-foreground">
            <span className="text-gradient">Our Analysis Process</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed font-light">
            Discover how our AI-powered system analyzes your facial features using advanced computer vision and mathematical principles
          </p>
        </motion.div>

        {/* Process Steps */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-20">
          {steps.map((step, index) => {
            const Icon = step.icon
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card className="glass-panel border-black/5 h-full hover:border-primary/30 transition-all duration-300">
                  <CardHeader>
                    <div className={`h-16 w-16 rounded-2xl bg-gradient-to-br ${step.color} flex items-center justify-center mb-4 shadow-[0_0_15px_rgba(255,215,0,0.5)]`}>
                      <Icon className="h-8 w-8 text-foreground" />
                    </div>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-sm font-bold text-primary">Step {index + 1}</span>
                    </div>
                    <CardTitle className="text-xl text-foreground">{step.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-muted-foreground leading-relaxed">
                      {step.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              </motion.div>
            )
          })}
        </div>

        {/* Features Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-center mb-12 text-foreground">
            Advanced <span className="text-gradient">Technology</span>
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, delay: 0.4 + index * 0.1 }}
              >
                <Card className="glass-panel border-black/5 hover:border-primary/30 transition-all duration-300">
                  <CardContent className="p-6">
                    <div className="flex items-start gap-4">
                      <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                        <CheckCircle2 className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-foreground mb-2">{feature.title}</h3>
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {feature.description}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Technical Details */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="glass-panel border-black/5 rounded-3xl p-8 md:p-12 mb-12"
        >
          <div className="grid md:grid-cols-2 gap-12">
            <div>
              <div className="flex items-center gap-3 mb-6">
                <Zap className="h-6 w-6 text-primary" />
                <h3 className="text-2xl font-bold text-foreground">How It Works</h3>
              </div>
              <div className="space-y-4 text-muted-foreground">
                <p className="leading-relaxed">
                  Our system uses <strong className="text-foreground">MediaPipe Face Landmarker</strong>, a state-of-the-art AI model that detects 468 facial landmarks in real-time.
                </p>
                <p className="leading-relaxed">
                  These landmarks are then analyzed using <strong className="text-foreground">geometric calculations</strong> to measure:
                </p>
                <ul className="space-y-2 ml-4">
                  <li className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span>Facial symmetry (eye, nose, mouth alignment)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span>Proportional ratios (golden ratio, facial thirds)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span>Feature balance (eye spacing, nose width, mouth width)</span>
                  </li>
                </ul>
              </div>
            </div>
            <div>
              <div className="flex items-center gap-3 mb-6">
                <Shield className="h-6 w-6 text-primary" />
                <h3 className="text-2xl font-bold text-foreground">Privacy & Security</h3>
              </div>
              <div className="space-y-4 text-muted-foreground">
                <p className="leading-relaxed">
                  All processing happens <strong className="text-foreground">entirely in your browser</strong>. Your images are never uploaded to our servers.
                </p>
                <p className="leading-relaxed">
                  The analysis is performed using <strong className="text-foreground">client-side AI</strong>, meaning:
                </p>
                <ul className="space-y-2 ml-4">
                  <li className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span>Images are processed in memory only</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span>No permanent storage of your photos</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span>Automatic cleanup when you close the page</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="text-center"
        >
          <Link href="/dashboard">
            <Button size="lg" className="bg-gradient-to-r from-primary to-secondary hover:opacity-90 text-black font-semibold shadow-[0_0_20px_-5px_hsl(45,100%,50%,0.5)] text-lg px-8 py-7 rounded-2xl">
              Try It Now
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
        </motion.div>
      </div>
    </div>
  )
}
