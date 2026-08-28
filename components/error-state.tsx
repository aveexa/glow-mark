"use client"

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ERROR_MESSAGES, ErrorCode } from '@/lib/constants'
import { AlertCircle, RefreshCw } from 'lucide-react'

interface ErrorStateProps {
  errorCode: ErrorCode
  /** Specific text from the backend, e.g. "Please close your mouth" for
   *  EXPRESSION_NOT_NEUTRAL. Shown instead of the generic per-code message. */
  hint?: string | null
  onRetry?: () => void
  onCancel?: () => void
}

export function ErrorState({
  errorCode,
  hint,
  onRetry,
  onCancel,
}: ErrorStateProps) {
  const message = ERROR_MESSAGES[errorCode]
  const detail = hint?.trim() || null

  return (
    <Card className="border-2 border-red-300 shadow-2xl bg-gradient-to-br from-white to-red-50/30 animate-scale-in">
      <CardHeader className="bg-gradient-to-r from-red-500 to-red-600 text-foreground rounded-t-lg">
        <CardTitle className="flex items-center gap-3 text-xl">
          <AlertCircle className="h-6 w-6" />
          Analysis Error
        </CardTitle>
      </CardHeader>
      <CardContent className="p-8 space-y-6">
        <div className="p-6 bg-red-50 border-2 border-red-200 rounded-xl space-y-2">
          <p className="text-base text-gray-800 font-medium leading-relaxed">{detail ?? message}</p>
          {detail && (
            <p className="text-sm text-gray-600 leading-relaxed">{message}</p>
          )}
        </div>

        <div className="flex flex-col sm:flex-row gap-3">
          {onRetry && (
            <Button
              onClick={onRetry}
              className="flex-1 bg-gradient-to-r from-yellow-600 to-amber-600 hover:from-yellow-700 hover:to-amber-700 shadow-lg hover:shadow-xl transition-all duration-300"
              size="lg"
            >
              <RefreshCw className="mr-2 h-5 w-5" />
              Try Again
            </Button>
          )}
          {onCancel && (
            <Button
              onClick={onCancel}
              variant="outline"
              className="flex-1 border-2 hover:bg-gray-50 transition-all duration-300"
              size="lg"
            >
              Go Back
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
