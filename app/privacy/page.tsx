import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { ArrowLeft, Shield } from 'lucide-react'

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-background py-16 text-foreground relative overflow-hidden">
      {/* Immersive Background */}
      <div className="absolute inset-0 -z-10 bg-background overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-[120px] mix-blend-screen" />
        <div className="absolute bottom-1/4 right-1/4 w-[28rem] h-[28rem] bg-secondary/10 rounded-full blur-[120px] mix-blend-screen" />
        <div className="absolute inset-0 bg-grid-gold/[0.02]" />
      </div>

      <div className="container mx-auto px-4 max-w-4xl relative z-10">
        <Link href="/">
          <Button variant="ghost" className="mb-6 hover:bg-black/5 text-muted-foreground hover:text-foreground transition-colors">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Hub
          </Button>
        </Link>

        <Card className="bg-black/40 border border-black/5 shadow-[0_0_50px_-15px_rgba(255,215,0,0.15)] glass-panel overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-primary/20 via-black to-transparent border-b border-black/5">
            <CardTitle className="text-4xl flex items-center gap-3 text-foreground">
              <Shield className="h-8 w-8 text-primary" />
              Privacy Policy
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-8 p-8 max-w-none">

            <section className="bg-black/5 p-6 rounded-xl border border-black/5">
              <h2 className="text-2xl font-bold mb-4 text-primary">Data Privacy Commitment</h2>
              <p className="text-muted-foreground leading-relaxed">
                At GlowMark Aesthetics, we take your privacy seriously. This policy explains
                how we handle your photos and beauty analysis data with complete transparency.
              </p>
            </section>

            <section className="bg-black/5 p-6 rounded-xl border border-black/5">
              <h2 className="text-2xl font-bold mb-4 text-primary">Photo Processing</h2>
              <ul className="space-y-3 text-muted-foreground">
                <li className="flex items-start gap-3">
                  <div className="h-2 w-2 rounded-full bg-primary mt-2 flex-shrink-0 shadow-[0_0_10px_rgba(255,215,0,1)]"></div>
                  <div>
                    <strong className="text-foreground">Never Stored:</strong> Photos are processed
                    instantly for your beauty profile and are never saved to our disks or databases.
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <div className="h-2 w-2 rounded-full bg-secondary mt-2 flex-shrink-0 shadow-[0_0_10px_rgba(255,215,0,1)]"></div>
                  <div>
                    <strong className="text-foreground">Instant Deletion:</strong> All facial analysis happens
                    in real-time and your data is permanently deleted the moment your results are ready.
                  </div>
                </li>
              </ul>
            </section>

            <section className="bg-black/5 p-6 rounded-xl border border-black/5">
              <h2 className="text-2xl font-bold mb-4 text-primary">Session Security</h2>
              <p className="text-muted-foreground mb-3 leading-relaxed">
                Your consultation results are completely private. This data is purged:
              </p>
              <ul className="space-y-2 text-muted-foreground">
                <li className="flex items-center gap-2">
                  <span className="text-primary font-bold">•</span>
                  <span>When navigating away from the results page</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-primary font-bold">•</span>
                  <span>When you manually click &quot;Clear Results&quot;</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-primary font-bold">•</span>
                  <span>Upon closing your browser</span>
                </li>
              </ul>
            </section>

            <div className="pt-6 border-t border-black/5">
              <Link href="/">
                <Button className="bg-gradient-to-r from-primary to-secondary hover:opacity-90 shadow-[0_0_20px_-5px_hsl(45,100%,50%,0.5)] border border-black/5 text-black font-semibold">
                  Acknowledge & Return
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
