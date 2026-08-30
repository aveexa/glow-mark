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
  category?: string;
  severity?: string;
  trigger_class?: string;
}

/** One selectable comparison group. `value` is the backend's internal region key. */
export interface RegionChoice {
  value: string;
  label: string;
}

/** Which comparison group the measurements were scored against.
 *  A reference population, never an identity claim about the user. */
export interface RegionInfo {
  /** Full 7-way mixture, or null when no group was used (global arm). */
  weights: Record<string, number> | null;
  /** Human-readable group name, e.g. "South Asian" or "South Asian / Middle Eastern". */
  reference_label: string | null;
  source: 'inferred' | 'user_override' | 'global_fallback';
  overridable: boolean;
  /** The pickable groups, owned by backend/region.py so no client copy can drift. */
  choices: RegionChoice[];
  /** Which choice the picker should show as selected. */
  selected: string | null;
}

/** Gate readings for this request, surfaced for display and debugging. */
export interface GateResults {
  pose: { yaw_deg: number; pitch_deg: number; roll_deg: number };
  realness: { p_photo: number };
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
    /** [p20, p80] for the comparison group — what is TYPICAL, not what is ideal.
     *  null when the reference table has no cell for this feature, and absent
     *  entirely on analyses stored before the field existed. */
    typicalRange?: [number, number] | null;
  }>;
  /** Legacy Feature MLP non-ok strings (e.g. "nose_width_ratio: high"). */
  recommendations: string[];
  /** Structured Feature MLP output for all 24 geometry labels. */
  recommendation_items?: RecommendationItem[];
  /** Catalog tips from suggestion ranker (Brain C). */
  suggestions?: CatalogSuggestion[];
  /** One-paragraph plain-language summary of the suggestions above. */
  summary?: string;
  /** Comparison group these measurements were scored against. */
  region?: RegionInfo;
  gates?: GateResults;
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
    /** [p20, p80] for the comparison group — what is TYPICAL, not what is ideal.
     *  null when the reference table has no cell for this feature. */
    typicalRange: [number, number] | null;
  }>;
  recommendations: string[];
  recommendation_items?: RecommendationItem[];
  suggestions?: CatalogSuggestion[];
  /** One-paragraph plain-language summary of the suggestions, built from approved
   *  catalog text. Empty string when there is nothing to summarise. */
  summary?: string;
  region?: RegionInfo;
  gates?: GateResults;
  notes?: string[];
}

export interface BackendAnalyzeErrorResponse {
  error: ErrorCode | string;
  details?: string;
  /** Specific, actionable text from the backend gates, e.g. "Please close your mouth".
   *  Preferred over the generic per-code message when present. */
  hint?: string;
}

export type BackendAnalyzeApiResponse = BackendAnalyzeResponse | BackendAnalyzeErrorResponse;

export interface AnalysisState {
  selectedFile: File | null;
  previewUrl: string | null;
  analysisStatus: 'idle' | 'uploading' | 'processing' | 'success' | 'error';
  progressStep: ProcessingStep | null;
  result: AnalysisResult | null;
  error: ErrorCode | null;
  /** Backend hint attached to the current error, when it sent one. */
  errorHint: string | null;
}
