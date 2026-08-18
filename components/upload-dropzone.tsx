"use client"

import { useCallback, useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { uploadSchema } from '@/lib/schemas'
import { ACCEPTED_IMAGE_TYPES, MAX_FILE_SIZE } from '@/lib/constants'
import { Upload, X, FileImage, Sparkles, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { motion, AnimatePresence } from 'framer-motion'

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void
  disabled?: boolean
}

export function UploadDropzone({ onFileSelect, disabled }: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const {
    register,
    formState: { errors },
    setValue,
  } = useForm<{ file: File }>({
    resolver: zodResolver(uploadSchema),
  })

  // We type the file input explicitly
  const fileInputRef = register('file').ref

  const validateFile = (file: File): string | null => {
    if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
      return 'Invalid format. Use JPEG, PNG, or WebP.'
    }
    if (file.size > MAX_FILE_SIZE) {
      return 'File exceeds the 5MB limit.'
    }
    return null
  }

  const handleFile = useCallback(
    (file: File) => {
      const validationError = validateFile(file)
      if (validationError) {
        setError(validationError)
        return
      }

      setError(null)
      setValue('file', file)
      onFileSelect(file)
    },
    [onFileSelect, setValue]
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!disabled) setIsDragging(true)
  }, [disabled])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragging(false)

      if (disabled) return

      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile, disabled]
  )

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const displayError = error || errors.file?.message

  return (
    <div className="w-full relative z-10">
      <motion.div
        className="relative group"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Glow effect matching dragging state */}
        <div className={cn(
          "absolute -inset-1 blur-2xl transition-all duration-700 opacity-20",
          isDragging ? "bg-gradient-to-r from-primary to-accent opacity-50" : "bg-primary/20 group-hover:bg-primary/40 group-hover:opacity-40"
        )} />

        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            'relative rounded-[2rem] p-12 md:p-16 transition-all duration-500 ease-out flex flex-col items-center justify-center text-center',
            'glass-panel overflow-hidden',
            isDragging && !disabled
              ? 'border-primary/50 shadow-[0_0_40px_-10px_rgba(255,215,0,0.5)] scale-[1.02]'
              : 'border-black/5 hover:border-white/20',
            disabled && 'opacity-50 cursor-not-allowed select-none'
          )}
        >
          {/* Subtle dash border overlay */}
          <div className="absolute inset-4 rounded-[1.5rem] border-2 border-dashed border-black/5 pointer-events-none" />

          <input
            {...register('file')}
            ref={fileInputRef}
            type="file"
            // Ensure inputs don't bubble events poorly
            onClick={(e) => { e.stopPropagation() }}
            accept={ACCEPTED_IMAGE_TYPES.join(',')}
            onChange={handleFileInput}
            disabled={disabled}
            className="hidden"
            aria-label="Upload image"
          />

          <AnimatePresence mode="wait">
            <motion.div
              key={isDragging ? 'dragging' : 'idle'}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="relative flex flex-col items-center justify-center z-10"
            >
              <div className={cn(
                'h-24 w-24 rounded-2xl flex items-center justify-center mb-8 relative transition-all duration-500',
                isDragging
                  ? 'bg-gradient-to-br from-primary via-amber-500 to-accent shadow-[0_0_30px_rgba(255,215,0,0.6)]'
                  : 'bg-black/5 group-hover:bg-black/5 group-hover:scale-105 shadow-[0_0_15px_rgba(255,255,255,0.05)] group-hover:bg-primary/10'
              )}>
                {isDragging ? (
                  <Sparkles className="h-10 w-10 text-foreground animate-pulse" />
                ) : (
                  <div className="relative">
                    <FileImage className="h-10 w-10 text-muted-foreground group-hover:text-primary transition-colors" />
                    <Upload className="h-5 w-5 text-muted-foreground/50 absolute -top-2 -right-2 group-hover:text-primary/70 transition-colors" />
                  </div>
                )}
                <div className="absolute inset-0 bg-white/20 blur-xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>

              <h3 className="text-2xl font-bold text-foreground mb-3">
                {isDragging ? 'Release to Begin' : 'Upload for Analysis'}
              </h3>

              <div className="text-muted-foreground max-w-sm mb-8 text-[15px] font-light">
                Drag and drop your high-resolution image, or{' '}
                <button
                  type="button"
                  onClick={() => !disabled && document.querySelector('input[type="file"]')?.dispatchEvent(new MouseEvent('click'))}
                  disabled={disabled}
                  className="text-primary hover:text-foreground font-medium hover:underline underline-offset-4 disabled:cursor-not-allowed transition-all relative z-20"
                >
                  browse your device
                </button>
              </div>

              <div className="flex flex-wrap items-center justify-center gap-4">
                <div className="px-4 py-2 bg-black/5 rounded-full border border-black/5 backdrop-blur-md">
                  <span className="text-xs font-medium text-foreground/70">JPEG, PNG, WebP</span>
                </div>
                <div className="px-4 py-2 bg-black/5 rounded-full border border-black/5 backdrop-blur-md">
                  <span className="text-xs font-medium text-foreground/70">Max 5MB Focus</span>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </motion.div>

      {/* Error State */}
      <AnimatePresence>
        {displayError && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-6 flex items-start gap-4 text-sm bg-destructive/10 border border-destructive/30 rounded-2xl p-4 backdrop-blur-md"
          >
            <div className="h-10 w-10 rounded-xl bg-destructive/20 flex items-center justify-center flex-shrink-0">
              <AlertCircle className="h-5 w-5 text-destructive" />
            </div>
            <div className="flex-1">
              <p className="font-semibold text-foreground mb-1">Validation Failed</p>
              <p className="text-destructive-foreground">{displayError}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
