import { Mail, MessageSquare, MapPin } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export default function ContactPage() {
    return (
        <div className="min-h-[calc(100vh-6rem)] bg-background text-foreground py-16 relative overflow-hidden flex items-center">
            <div className="absolute inset-0 -z-10 overflow-hidden">
                <div className="absolute bottom-0 right-0 w-[50rem] h-[50rem] bg-secondary/10 rounded-full blur-[150px] mix-blend-screen" />
                <div className="absolute inset-0 bg-grid-gold/[0.02]" />
            </div>

            <div className="container mx-auto px-4 max-w-6xl relative z-10">
                <div className="grid lg:grid-cols-2 gap-16 items-center">

                    <div>
                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-black/5 border border-black/5 text-secondary text-sm font-medium mb-6">
                            <MessageSquare className="h-4 w-4" />
                            Secure Contact
                        </div>
                        <h1 className="text-5xl md:text-6xl font-bold mb-6">
                            Get in <span className="text-gradient">Touch</span>
                        </h1>
                        <p className="text-xl text-muted-foreground leading-relaxed mb-12">
                            For personalized consultation inquiries, partnership opportunities, or general support, please send us a message below.
                        </p>

                        <div className="space-y-6">
                            <div className="flex items-center gap-4 text-muted-foreground">
                                <div className="h-12 w-12 rounded-xl bg-black/5 flex items-center justify-center border border-black/5">
                                    <Mail className="h-5 w-5 text-primary" />
                                </div>
                                <div>
                                    <div className="font-semibold text-foreground">Email Us</div>
                                    <div>hello@glowmark.style</div>
                                </div>
                            </div>
                            <div className="flex items-center gap-4 text-muted-foreground">
                                <div className="h-12 w-12 rounded-xl bg-black/5 flex items-center justify-center border border-black/5">
                                    <MapPin className="h-5 w-5 text-secondary" />
                                </div>
                                <div>
                                    <div className="font-semibold text-foreground">Our Studio</div>
                                    <div>Beverly Hills, California</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-black/40 border border-black/5 p-8 md:p-12 rounded-3xl glass-panel shadow-[0_0_50px_-15px_rgba(255,215,0,0.15)]">
                        <form className="space-y-6">
                            <div className="grid grid-cols-2 gap-6">
                                <div className="space-y-2">
                                    <Label htmlFor="firstName" className="text-muted-foreground">First Name</Label>
                                    <Input id="firstName" className="bg-black/5 border-black/5 text-foreground h-12" />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="lastName" className="text-muted-foreground">Last Name</Label>
                                    <Input id="lastName" className="bg-black/5 border-black/5 text-foreground h-12" />
                                </div>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="email" className="text-muted-foreground">Email Address</Label>
                                <Input id="email" type="email" className="bg-black/5 border-black/5 text-foreground h-12" />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="message" className="text-muted-foreground">Message</Label>
                                <textarea
                                    id="message"
                                    rows={4}
                                    className="flex w-full rounded-md border border-black/5 bg-black/5 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 resize-none"
                                ></textarea>
                            </div>
                            <Button className="w-full h-12 bg-gradient-to-r from-primary to-secondary hover:opacity-90 text-black font-semibold text-lg rounded-xl">
                                Send Message
                            </Button>
                        </form>
                    </div>

                </div>
            </div>
        </div>
    )
}
