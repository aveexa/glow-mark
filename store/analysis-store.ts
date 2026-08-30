import { create } from 'zustand';
import { AnalysisState, AnalysisResult } from '@/lib/types';
import { ErrorCode } from '@/lib/constants';
import { ProcessingStep } from '@/lib/constants';

interface AnalysisStore extends AnalysisState {
  setFile: (file: File | null) => void;
  setPreviewUrl: (url: string | null) => void;
  setAnalysisStatus: (status: AnalysisState['analysisStatus']) => void;
  setProgressStep: (step: ProcessingStep | null) => void;
  setResult: (result: AnalysisResult | null) => void;
  setError: (error: ErrorCode | null, hint?: string | null) => void;
  clearAll: () => void;
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  selectedFile: null,
  previewUrl: null,
  analysisStatus: 'idle',
  progressStep: null,
  result: null,
  error: null,
  errorHint: null,

  // Drop any previous result: it belongs to the previous photo, and leaving it in
  // place lets a stale overlay be drawn over a newly selected image.
  setFile: (file) => set({ selectedFile: file, result: null }),
  
  setPreviewUrl: (url) => {
    set((state) => {
      // Revoke previous URL if exists
      if (state.previewUrl) {
        URL.revokeObjectURL(state.previewUrl);
      }
      return { previewUrl: url };
    });
  },

  setAnalysisStatus: (status) => set({ analysisStatus: status }),

  setProgressStep: (step) => set({ progressStep: step }),

  setResult: (result) => set({ result, error: null, errorHint: null }),

  // Clearing an error is not an error state. Forcing status to 'error' here meant
  // every caller that did setError(null) before a request left the store claiming a
  // failure that had not happened, which broke the render branches downstream.
  setError: (error, hint = null) =>
    set(error === null
      ? { error: null, errorHint: null }
      : { error, errorHint: hint, analysisStatus: 'error' }),

  clearAll: () => {
    set((state) => {
      // Clean up object URL
      if (state.previewUrl) {
        URL.revokeObjectURL(state.previewUrl);
      }
      return {
        selectedFile: null,
        previewUrl: null,
        analysisStatus: 'idle',
        progressStep: null,
        result: null,
        error: null,
        errorHint: null,
      };
    });
  },
}));
