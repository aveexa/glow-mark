import { ErrorCode } from './constants';
import { ProcessingStep } from './constants';

export interface Landmark {
  x: number;
  y: number;
  z?: number; // Depth coordinate for 3D visualization
}

export interface CatalogSuggestion {
  id: string;
  text: string;
  confidence: number;
}

export interface RecommendationItem {
  label: string;
  class: string;
  confidence: number;
}

export interface AnalysisResult {
  score: number;
  metrics: {
    symmetry: number;
    proportions: number;
    balance: number;
  };
  landmarks: Landmark[];
  overlayTypeHints: {
    points: boolean;
    outline: boolean;
    mesh: boolean;
  };
  ratios: Array<{
    name: string;
    value: number;
    idealRange: string;
  }>;
  /** Legacy Feature MLP non-ok strings (e.g. "nose_width_ratio: high"). */
  recommendations: string[];
  /** Structured Feature MLP output for all 24 geometry labels. */
  recommendation_items?: RecommendationItem[];
  /** Catalog tips from suggestion ranker (Brain C). */
  suggestions?: CatalogSuggestion[];
  notes: string[];
}

/** Successful Flask POST /analyze JSON body. */
export interface BackendAnalyzeResponse {
  score: number;
  score_raw?: number;
  metrics: {
    symmetry: number;
    proportions: number;
    balance: number;
  };
  landmarks: Landmark[];
  overlayTypeHints: {
    points: boolean;
    outline: boolean;
    mesh: boolean;
  };
  ratios: Array<{
    name: string;
    value: number;
    idealRange: string;
  }>;
  recommendations: string[];
  recommendation_items?: RecommendationItem[];
  suggestions?: CatalogSuggestion[];
  notes?: string[];
}

export interface BackendAnalyzeErrorResponse {
  error: string;
  details?: string;
}

export type BackendAnalyzeApiResponse = BackendAnalyzeResponse | BackendAnalyzeErrorResponse;

export interface AnalysisState {
  selectedFile: File | null;
  previewUrl: string | null;
  analysisStatus: 'idle' | 'uploading' | 'processing' | 'success' | 'error';
  progressStep: ProcessingStep | null;
  result: AnalysisResult | null;
  error: ErrorCode | null;
}
