"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { motion } from "framer-motion"
import { Sparkles, User, LogIn, Menu, X, LogOut, Settings } from "lucide-react"
import { cn } from "@/lib/utils"
import { useState, useEffect } from "react"
import { Button } from "./ui/button"
import { useAuth } from "@/contexts/auth-context"
import { signOutUser } from "@/lib/firebase/auth"
import { useToast } from "@/hooks/use-toast"

const navLinks = [
    { href: "/", label: "Home" },
    { href: "/how-it-works", label: "How We Analyze" },
    { href: "/about", label: "About Us" },
    { href: "/contact", label: "Contact" }
]

export function Navbar() {
    const pathname = usePathname()
    const router = useRouter()
    const { user, userData, loading } = useAuth()
    const { toast } = useToast()
    const [isScrolled, setIsScrolled] = useState(false)
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
    const [userMenuOpen, setUserMenuOpen] = useState(false)

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 20)
        }
        window.addEventListener("scroll", handleScroll)
        return () => window.removeEventListener("scroll", handleScroll)
    }, [])

    // Close mobile menu when route changes
    useEffect(() => {
        setMobileMenuOpen(false)
    }, [pathname])

    // Close user menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (userMenuOpen && !(event.target as Element).closest('.user-menu-container')) {
                setUserMenuOpen(false)
            }
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [userMenuOpen])

    return (
        <motion.header
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
            className={cn(
                "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
                isScrolled ? "py-3" : "py-5"
            )}
        >
            <div className="container mx-auto px-4">
                <div className={cn(
                    "flex items-center justify-between px-6 py-3 rounded-2xl transition-all duration-500",
                    isScrolled
                        ? "glass-panel border-black/5 shadow-[0_8px_32px_rgba(0,0,0,0.6)]"
                        : "bg-transparent border-transparent"
                )}>
                    {/* Logo */}
                    <Link href="/" className="flex items-center gap-2 group relative z-50">
                        <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary via-amber-500 to-accent text-foreground shadow-[0_0_20px_rgba(255,215,0,0.4)] group-hover:shadow-[0_0_30px_rgba(255,215,0,0.6)] transition-all duration-300">
                            <Sparkles className="h-5 w-5 absolute" />
                            <div className="absolute inset-0 bg-white/20 rounded-xl blur-md opacity-0 group-hover:opacity-100 transition-opacity" />
                        </div>
                        <span className="text-xl font-bold tracking-tight text-foreground">
                            Glow<span className="text-primary group-hover:text-accent transition-colors duration-300">Mark</span>
                        </span>
                    </Link>

                    {/* Desktop Navigation */}
                    <nav className="hidden md:flex items-center gap-8">
                        <ul className="flex items-center gap-6">
                            {navLinks.map((link) => (
                                <li key={link.href}>
                                    <Link
                                        href={link.href}
                                        className={cn(
                                            "text-sm font-medium transition-colors hover:text-foreground relative py-2",
                                            pathname === link.href ? "text-foreground" : "text-muted-foreground"
                                        )}
                                    >
                                        {link.label}
                                        {pathname === link.href && (
                                            <motion.div
                                                layoutId="nav-indicator"
                                                className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary to-accent rounded-full shadow-[0_0_10px_rgba(255,215,0,0.5)]"
                                            />
                                        )}
                                    </Link>
                                </li>
                            ))}
                            {user && (
                                <li>
                                    <Link
                                        href="/dashboard"
                                        className={cn(
                                            "text-sm font-medium transition-colors hover:text-foreground relative py-2",
                                            pathname === "/dashboard" ? "text-foreground" : "text-muted-foreground"
                                        )}
                                    >
                                        Dashboard
                                        {pathname === "/dashboard" && (
                                            <motion.div
                                                layoutId="nav-indicator"
                                                className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary to-accent rounded-full shadow-[0_0_10px_rgba(255,215,0,0.5)]"
                                            />
                                        )}
                                    </Link>
                                </li>
                            )}
                        </ul>

                        <div className="h-6 w-px bg-black/5" />

                        {loading ? (
                            <div className="h-10 w-20 bg-black/5 rounded-xl animate-pulse" />
                        ) : user ? (
                            <div className="relative user-menu-container">
                                <button
                                    onClick={() => setUserMenuOpen(!userMenuOpen)}
                                    className="flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-black/5 transition-colors"
                                >
                                    {user.photoURL ? (
                                        <img
                                            src={user.photoURL}
                                            alt={user.displayName || "User"}
                                            className="h-8 w-8 rounded-full border-2 border-primary/30"
                                        />
                                    ) : (
                                        <div className="h-8 w-8 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center text-foreground font-semibold text-sm">
                                            {user.displayName?.[0] || user.email?.[0]?.toUpperCase() || "U"}
                                        </div>
                                    )}
                                    <span className="hidden lg:block text-sm font-medium text-foreground">
                                        {user.displayName || user.email?.split("@")[0] || "User"}
                                    </span>
                                </button>

                                {userMenuOpen && (
                                    <motion.div
                                        initial={{ opacity: 0, y: -10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="absolute right-0 mt-2 w-56 glass-panel border border-black/5 rounded-xl shadow-xl p-2 z-50"
                                    >
                                        <div className="px-3 py-2 border-b border-black/5 mb-2">
                                            <p className="text-sm font-semibold text-foreground">{user.displayName || "User"}</p>
                                            <p className="text-xs text-muted-foreground truncate">{user.email}</p>
                                        </div>
                                        <Link href="/dashboard" onClick={() => setUserMenuOpen(false)}>
                                            <button className="w-full text-left px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-black/5 rounded-lg transition-colors flex items-center gap-2">
                                                <User className="h-4 w-4" />
                                                Dashboard
                                            </button>
                                        </Link>
                                        <Link href="/dashboard" onClick={() => setUserMenuOpen(false)}>
                                            <button className="w-full text-left px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-black/5 rounded-lg transition-colors flex items-center gap-2">
                                                <Settings className="h-4 w-4" />
                                                Settings
                                            </button>
                                        </Link>
                                        <div className="h-px bg-black/5 my-2" />
                                        <button
                                            onClick={async () => {
                                                try {
                                                    await signOutUser()
                                                    toast({
                                                        title: "Signed out",
                                                        description: "You have been successfully signed out.",
                                                    })
                                                    setUserMenuOpen(false)
                                                    router.push("/")
                                                } catch (error) {
                                                    toast({
                                                        title: "Error",
                                                        description: "Failed to sign out. Please try again.",
                                                        variant: "destructive",
                                                    })
                                                }
                                            }}
                                            className="w-full text-left px-3 py-2 text-sm text-destructive hover:bg-destructive/10 rounded-lg transition-colors flex items-center gap-2"
                                        >
                                            <LogOut className="h-4 w-4" />
                                            Sign Out
                                        </button>
                                    </motion.div>
                                )}
                            </div>
                        ) : (
                            <div className="flex items-center gap-4">
                                <Link href="/login">
                                    <Button variant="ghost" className="text-muted-foreground hover:text-foreground hover:bg-black/5 rounded-xl">
                                        Sign In
                                    </Button>
                                </Link>
                                <Link href="/login?tab=register">
                                    <Button className="bg-gradient-to-r from-primary to-accent hover:opacity-90 shadow-[0_0_20px_-5px_hsl(45,100%,50%,0.5)] rounded-xl border border-black/5 text-black">
                                        <User className="mr-2 h-4 w-4" />
                                        Get Started
                                    </Button>
                                </Link>
                            </div>
                        )}
                    </nav>

                    {/* Mobile Menu Toggle */}
                    <button
                        className="md:hidden relative z-50 p-2 text-muted-foreground hover:text-foreground transition-colors"
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    >
                        {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                    </button>
                </div>
            </div>

            {/* Mobile Menu Dropdown */}
            <motion.div
                initial={false}
                animate={{
                    height: mobileMenuOpen ? "auto" : 0,
                    opacity: mobileMenuOpen ? 1 : 0
                }}
                className="md:hidden overflow-hidden bg-background/95 backdrop-blur-xl border-b border-black/5 absolute top-full left-0 right-0 shadow-2xl"
            >
                <div className="p-6 flex flex-col gap-6">
                    <ul className="flex flex-col gap-4">
                        {navLinks.map((link) => (
                            <li key={link.href}>
                                <Link
                                    href={link.href}
                                    className={cn(
                                        "text-lg font-medium block transition-colors",
                                        pathname === link.href ? "text-primary tracking-wide" : "text-muted-foreground"
                                    )}
                                >
                                    {link.label}
                                </Link>
                            </li>
                        ))}
                    </ul>

                    <div className="h-px w-full bg-black/5" />

                    {loading ? (
                        <div className="h-12 w-full bg-black/5 rounded-xl animate-pulse" />
                    ) : user ? (
                        <div className="flex flex-col gap-3">
                            <div className="flex items-center gap-3 px-3 py-2 bg-black/5 rounded-xl">
                                {user.photoURL ? (
                                    <img
                                        src={user.photoURL}
                                        alt={user.displayName || "User"}
                                        className="h-10 w-10 rounded-full border-2 border-primary/30"
                                    />
                                ) : (
                                    <div className="h-10 w-10 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center text-foreground font-semibold">
                                        {user.displayName?.[0] || user.email?.[0]?.toUpperCase() || "U"}
                                    </div>
                                )}
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm font-semibold text-foreground truncate">{user.displayName || "User"}</p>
                                    <p className="text-xs text-muted-foreground truncate">{user.email}</p>
                                </div>
                            </div>
                            <Link href="/dashboard" className="w-full">
                                <Button variant="outline" className="w-full justify-center glass-panel border-black/5 h-12 text-base rounded-xl">
                                    <User className="mr-2 h-4 w-4" />
                                    Dashboard
                                </Button>
                            </Link>
                            <Button
                                onClick={async () => {
                                    try {
                                        await signOutUser()
                                        toast({
                                            title: "Signed out",
                                            description: "You have been successfully signed out.",
                                        })
                                        router.push("/")
                                    } catch (error) {
                                        toast({
                                            title: "Error",
                                            description: "Failed to sign out. Please try again.",
                                            variant: "destructive",
                                        })
                                    }
                                }}
                                variant="outline"
                                className="w-full justify-center glass-panel border-destructive/30 text-destructive hover:bg-destructive/10 h-12 text-base rounded-xl"
                            >
                                <LogOut className="mr-2 h-4 w-4" />
                                Sign Out
                            </Button>
                        </div>
                    ) : (
                        <div className="flex flex-col gap-3">
                            <Link href="/login" className="w-full">
                                <Button variant="outline" className="w-full justify-center glass-panel border-black/5 h-12 text-base rounded-xl">
                                    <LogIn className="mr-2 h-4 w-4" />
                                    Sign In
                                </Button>
                            </Link>
                            <Link href="/login?tab=register" className="w-full">
                                <Button className="w-full justify-center bg-gradient-to-r from-primary to-accent shadow-[0_0_20px_-5px_hsl(45,100%,50%,0.5)] border border-black/5 h-12 text-base rounded-xl text-black">
                                    <User className="mr-2 h-4 w-4" />
                                    Create Account
                                </Button>
                            </Link>
                        </div>
                    )}
                </div>
            </motion.div>
        </motion.header>
    )
}
