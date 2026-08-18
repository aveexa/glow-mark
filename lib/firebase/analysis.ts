import { collection, addDoc, query, where, orderBy, getDocs, doc, getDoc, serverTimestamp, Timestamp } from 'firebase/firestore';
import { db } from './config';
import { AnalysisResult } from '@/lib/types';

export interface AnalysisRecord {
  id?: string;
  userId: string;
  result: AnalysisResult;
  // Note: We do NOT store images - only analysis results (scores, metrics, landmarks, recommendations)
  createdAt: Timestamp | Date;
  updatedAt: Timestamp | Date;
}

/**
 * Save analysis result to Firestore
 * Note: Images are NOT saved - only analysis data (scores, metrics, landmarks, recommendations)
 */
export async function saveAnalysisResult(
  userId: string,
  result: AnalysisResult
): Promise<string> {
  try {
    // Prepare data for Firestore (exclude any image data)
    const analysisData = {
      userId,
      score: result.score,
      metrics: {
        symmetry: result.metrics.symmetry,
        proportions: result.metrics.proportions,
        balance: result.metrics.balance,
      },
      ratios: result.ratios || [],
      recommendations: result.recommendations || [],
      recommendation_items: (result.recommendation_items || [])
        .filter((it) => it.class !== 'ok')
        .map((it) => ({
          label: it.label,
          class: it.class,
          confidence: it.confidence,
        })),
      suggestions: (result.suggestions || []).map((s) => ({
        id: s.id,
        text: s.text,
        confidence: s.confidence,
      })),
      notes: result.notes || [],
      // Store landmarks count but not the full array (to save space)
      landmarkCount: result.landmarks?.length || 0,
      overlayTypeHints: result.overlayTypeHints || {
        points: true,
        outline: true,
        mesh: false,
      },
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    };

    const docRef = await addDoc(collection(db, 'analyses'), analysisData);
    console.log('Analysis result saved successfully with ID:', docRef.id);
    return docRef.id;
  } catch (error) {
    console.error('Error saving analysis result:', error);
    throw error;
  }
}

/**
 * Get user's analysis history
 */
export async function getUserAnalyses(userId: string): Promise<AnalysisRecord[]> {
  try {
    // Try with orderBy first, fallback to without if index missing
    let q = query(
      collection(db, 'analyses'),
      where('userId', '==', userId),
      orderBy('createdAt', 'desc')
    );

    let querySnapshot;
    try {
      querySnapshot = await getDocs(q);
    } catch (indexError: any) {
      // If index error, try without orderBy
      if (indexError?.code === 'failed-precondition' || indexError?.message?.includes('index')) {
        console.warn('Firestore index missing, fetching without orderBy');
        const q2 = query(
          collection(db, 'analyses'),
          where('userId', '==', userId)
        );
        querySnapshot = await getDocs(q2);
      } else {
        throw indexError;
      }
    }
    const analyses: AnalysisRecord[] = [];
    const docsArray = Array.from(querySnapshot.docs);

    // Sort manually if orderBy was not used
    if (docsArray.length > 0) {
      docsArray.sort((a, b) => {
        const aData = a.data();
        const bData = b.data();
        const aTime = aData.createdAt?.toDate ? aData.createdAt.toDate() : (aData.createdAt instanceof Date ? aData.createdAt : new Date(aData.createdAt || 0));
        const bTime = bData.createdAt?.toDate ? bData.createdAt.toDate() : (bData.createdAt instanceof Date ? bData.createdAt : new Date(bData.createdAt || 0));
        return bTime.getTime() - aTime.getTime();
      });
    }

    docsArray.forEach((doc) => {
      const data = doc.data();
      // Reconstruct AnalysisResult from stored data
      const result: AnalysisResult = {
        score: data.score || data.result?.score || 0,
        metrics: data.metrics || data.result?.metrics || {
          symmetry: 0,
          proportions: 0,
          balance: 0,
        },
        landmarks: [], // Landmarks not stored to save space
        overlayTypeHints: data.overlayTypeHints || data.result?.overlayTypeHints || {
          points: true,
          outline: true,
          mesh: false,
        },
        ratios: data.ratios || data.result?.ratios || [],
        recommendations: data.recommendations || data.result?.recommendations || [],
        notes: data.notes || data.result?.notes || [],
      };
      
      analyses.push({
        id: doc.id,
        userId: data.userId,
        result,
        createdAt: data.createdAt?.toDate ? data.createdAt.toDate() : (data.createdAt instanceof Date ? data.createdAt : new Date(data.createdAt)),
        updatedAt: data.updatedAt?.toDate ? data.updatedAt.toDate() : (data.updatedAt instanceof Date ? data.updatedAt : new Date(data.updatedAt)),
      });
    });

    return analyses;
  } catch (error) {
    console.error('Error fetching user analyses:', error);
    return [];
  }
}

/**
 * Get a specific analysis by ID
 */
export async function getAnalysisById(analysisId: string): Promise<AnalysisRecord | null> {
  try {
    const docRef = doc(db, 'analyses', analysisId);
    const docSnap = await getDoc(docRef);

    if (docSnap.exists()) {
      const data = docSnap.data();
      // Reconstruct AnalysisResult from stored data
      const result: AnalysisResult = {
        score: data.score || data.result?.score || 0,
        metrics: data.metrics || data.result?.metrics || {
          symmetry: 0,
          proportions: 0,
          balance: 0,
        },
        landmarks: [], // Landmarks not stored to save space
        overlayTypeHints: data.overlayTypeHints || data.result?.overlayTypeHints || {
          points: true,
          outline: true,
          mesh: false,
        },
        ratios: data.ratios || data.result?.ratios || [],
        recommendations: data.recommendations || data.result?.recommendations || [],
        notes: data.notes || data.result?.notes || [],
      };
      
      return {
        id: docSnap.id,
        userId: data.userId,
        result,
        createdAt: data.createdAt?.toDate() || new Date(),
        updatedAt: data.updatedAt?.toDate() || new Date(),
      };
    }
    return null;
  } catch (error) {
    console.error('Error fetching analysis:', error);
    return null;
  }
}

/**
 * Calculate average metrics from user's analyses
 */
export async function getUserAverageMetrics(userId: string): Promise<{
  averageScore: number;
  averageSymmetry: number;
  averageProportions: number;
  averageBalance: number;
  totalAnalyses: number;
}> {
  try {
    const analyses = await getUserAnalyses(userId);
    
    if (analyses.length === 0) {
      return {
        averageScore: 0,
        averageSymmetry: 0,
        averageProportions: 0,
        averageBalance: 0,
        totalAnalyses: 0,
      };
    }

    const totals = analyses.reduce(
      (acc, analysis) => {
        acc.score += analysis.result.score;
        acc.symmetry += analysis.result.metrics.symmetry;
        acc.proportions += analysis.result.metrics.proportions;
        acc.balance += analysis.result.metrics.balance;
        return acc;
      },
      { score: 0, symmetry: 0, proportions: 0, balance: 0 }
    );

    const count = analyses.length;

    return {
      averageScore: Math.round(totals.score / count),
      averageSymmetry: Math.round(totals.symmetry / count),
      averageProportions: Math.round(totals.proportions / count),
      averageBalance: Math.round(totals.balance / count),
      totalAnalyses: count,
    };
  } catch (error) {
    console.error('Error calculating average metrics:', error);
    return {
      averageScore: 0,
      averageSymmetry: 0,
      averageProportions: 0,
      averageBalance: 0,
      totalAnalyses: 0,
    };
  }
}
