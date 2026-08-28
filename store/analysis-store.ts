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

  setFile: (file) => set({ selectedFile: file }),
  
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

  setError: (error, hint = null) => set({ error, errorHint: hint, analysisStatus: 'error' }),

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
