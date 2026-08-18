import { Landmark } from '@/lib/types';

/**
 * Calculate Euclidean distance between two points
 */
function distance(p1: { x: number; y: number }, p2: { x: number; y: number }): number {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

/**
 * Calculate midpoint between two points
 */
function midpoint(p1: { x: number; y: number }, p2: { x: number; y: number }): { x: number; y: number } {
  return {
    x: (p1.x + p2.x) / 2,
    y: (p1.y + p2.y) / 2
  };
}

/**
 * MediaPipe Face Landmark indices (468 landmarks)
 * Using approximate indices - will calculate centers dynamically if indices don't exist
 */
const LANDMARK_INDICES = {
  // Face outline (approximate)
  FACE_TOP: 10,
  FACE_BOTTOM: 152,
  FACE_LEFT: 234,
  FACE_RIGHT: 454,
  FACE_CENTER: 1,
  
  // Left eye region (indices 33-133 approximate)
  LEFT_EYE_REGION_START: 33,
  LEFT_EYE_REGION_END: 133,
  
  // Right eye region (indices 362-263 approximate)
  RIGHT_EYE_REGION_START: 362,
  RIGHT_EYE_REGION_END: 263,
  
  // Nose
  NOSE_TIP: 4,
  NOSE_BRIDGE: 6,
  NOSE_LEFT: 131,
  NOSE_RIGHT: 360,
  
  // Mouth
  MOUTH_LEFT: 61,
  MOUTH_RIGHT: 291,
  MOUTH_TOP: 13,
  MOUTH_BOTTOM: 14,
};

/**
 * Get safe landmark or return default
 */
function getLandmark(landmarks: Landmark[], index: number, fallback?: Landmark): Landmark {
  if (landmarks[index]) {
    return landmarks[index];
  }
  return fallback || { x: 0.5, y: 0.5 };
}

/**
 * Calculate center point of a region
 */
function calculateRegionCenter(landmarks: Landmark[], startIdx: number, endIdx: number): Landmark {
  const regionPoints: Landmark[] = [];
  for (let i = startIdx; i <= endIdx && i < landmarks.length; i++) {
    if (landmarks[i]) {
      regionPoints.push(landmarks[i]);
    }
  }
  
  if (regionPoints.length === 0) {
    return { x: 0.5, y: 0.5 };
  }
  
  const sum = regionPoints.reduce(
    (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }),
    { x: 0, y: 0 }
  );
  
  return {
    x: sum.x / regionPoints.length,
    y: sum.y / regionPoints.length,
  };
}

/**
 * Calculate facial symmetry score (0-100)
 */
function calculateSymmetry(landmarks: Landmark[]): number {
  if (landmarks.length < 100) return 50; // Need minimum landmarks
  
  // Calculate eye centers using region averaging
  const leftEyeCenter = calculateRegionCenter(
    landmarks,
    LANDMARK_INDICES.LEFT_EYE_REGION_START,
    LANDMARK_INDICES.LEFT_EYE_REGION_END
  );
  
  const rightEyeCenter = calculateRegionCenter(
    landmarks,
    LANDMARK_INDICES.RIGHT_EYE_REGION_START,
    LANDMARK_INDICES.RIGHT_EYE_REGION_END
  );
  
  // Vertical alignment of eyes
  const eyeVerticalDiff = Math.abs(leftEyeCenter.y - rightEyeCenter.y);
  const eyeSymmetry = Math.max(0, 100 - (eyeVerticalDiff * 2000));
  
  // Face center (use nose bridge or approximate center)
  const faceCenterX = getLandmark(landmarks, LANDMARK_INDICES.FACE_CENTER, { x: 0.5, y: 0.5 }).x;
  
  // Nose alignment
  const noseTip = getLandmark(landmarks, LANDMARK_INDICES.NOSE_TIP);
  const noseAlignment = Math.max(0, 100 - (Math.abs(noseTip.x - faceCenterX) * 400));
  
  // Mouth alignment
  const mouthLeft = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_LEFT);
  const mouthRight = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_RIGHT);
  const mouthCenter = midpoint(mouthLeft, mouthRight);
  const mouthAlignment = Math.max(0, 100 - (Math.abs(mouthCenter.x - faceCenterX) * 400));
  
  // Average symmetry scores
  return Math.round((eyeSymmetry * 0.4 + noseAlignment * 0.3 + mouthAlignment * 0.3));
}

/**
 * Calculate facial proportions score (0-100)
 */
function calculateProportions(landmarks: Landmark[]): number {
  if (landmarks.length < 100) return 50;
  
  // Get face boundaries
  const faceTop = getLandmark(landmarks, LANDMARK_INDICES.FACE_TOP);
  const faceBottom = getLandmark(landmarks, LANDMARK_INDICES.FACE_BOTTOM);
  const faceLeft = getLandmark(landmarks, LANDMARK_INDICES.FACE_LEFT);
  const faceRight = getLandmark(landmarks, LANDMARK_INDICES.FACE_RIGHT);
  
  const faceWidth = distance(faceLeft, faceRight);
  const faceHeight = distance(faceTop, faceBottom);
  
  if (faceHeight === 0) return 50;
  
  const faceRatio = faceWidth / faceHeight;
  
  // Ideal face ratio is around 0.618 (golden ratio) to 0.7
  const idealRatio = 0.65;
  const ratioScore = Math.max(0, 100 - (Math.abs(faceRatio - idealRatio) * 300));
  
  // Facial thirds
  const noseTip = getLandmark(landmarks, LANDMARK_INDICES.NOSE_TIP);
  const leftEyeCenter = calculateRegionCenter(
    landmarks,
    LANDMARK_INDICES.LEFT_EYE_REGION_START,
    LANDMARK_INDICES.LEFT_EYE_REGION_END
  );
  
  const upperThird = Math.abs(faceTop.y - leftEyeCenter.y);
  const middleThird = Math.abs(leftEyeCenter.y - noseTip.y);
  const lowerThird = Math.abs(noseTip.y - faceBottom.y);
  
  const totalHeight = upperThird + middleThird + lowerThird;
  if (totalHeight === 0) return Math.round(ratioScore);
  
  const upperRatio = upperThird / totalHeight;
  const middleRatio = middleThird / totalHeight;
  const lowerRatio = lowerThird / totalHeight;
  
  // Ideal thirds are approximately 0.33 each
  const idealThird = 0.33;
  const thirdsScore = Math.max(0, 100 - (
    Math.abs(upperRatio - idealThird) * 400 +
    Math.abs(middleRatio - idealThird) * 400 +
    Math.abs(lowerRatio - idealThird) * 400
  ));
  
  return Math.round((ratioScore * 0.5 + thirdsScore * 0.5));
}

/**
 * Calculate facial balance score (0-100)
 */
function calculateBalance(landmarks: Landmark[]): number {
  if (landmarks.length < 100) return 50;
  
  // Calculate eye regions
  const leftEyeRegion = calculateRegionCenter(
    landmarks,
    LANDMARK_INDICES.LEFT_EYE_REGION_START,
    LANDMARK_INDICES.LEFT_EYE_REGION_END
  );
  
  const rightEyeRegion = calculateRegionCenter(
    landmarks,
    LANDMARK_INDICES.RIGHT_EYE_REGION_START,
    LANDMARK_INDICES.RIGHT_EYE_REGION_END
  );
  
  // Approximate eye width using region spread
  const leftEyePoints: Landmark[] = [];
  for (let i = LANDMARK_INDICES.LEFT_EYE_REGION_START; i <= LANDMARK_INDICES.LEFT_EYE_REGION_END && i < landmarks.length; i++) {
    if (landmarks[i]) leftEyePoints.push(landmarks[i]);
  }
  
  let eyeWidth = 0.1; // Default
  if (leftEyePoints.length > 1) {
    const minX = Math.min(...leftEyePoints.map(p => p.x));
    const maxX = Math.max(...leftEyePoints.map(p => p.x));
    eyeWidth = maxX - minX;
  }
  
  // Eye spacing
  const eyeSpacing = distance(leftEyeRegion, rightEyeRegion);
  const eyeSpacingRatio = eyeWidth > 0 ? eyeSpacing / eyeWidth : 1.0;
  
  // Ideal eye spacing is approximately 1 eye width (ratio ~1.0)
  const idealEyeSpacing = 1.0;
  const eyeSpacingScore = Math.max(0, 100 - (Math.abs(eyeSpacingRatio - idealEyeSpacing) * 100));
  
  // Face width
  const faceLeft = getLandmark(landmarks, LANDMARK_INDICES.FACE_LEFT);
  const faceRight = getLandmark(landmarks, LANDMARK_INDICES.FACE_RIGHT);
  const faceWidth = distance(faceLeft, faceRight);
  
  if (faceWidth === 0) return Math.round(eyeSpacingScore);
  
  // Nose width relative to face width
  const noseLeft = getLandmark(landmarks, LANDMARK_INDICES.NOSE_LEFT);
  const noseRight = getLandmark(landmarks, LANDMARK_INDICES.NOSE_RIGHT);
  const noseWidth = distance(noseLeft, noseRight);
  const noseRatio = noseWidth / faceWidth;
  
  // Ideal nose width is approximately 0.25-0.3 of face width
  const idealNoseRatio = 0.275;
  const noseScore = Math.max(0, 100 - (Math.abs(noseRatio - idealNoseRatio) * 400));
  
  // Mouth width relative to face
  const mouthLeft = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_LEFT);
  const mouthRight = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_RIGHT);
  const mouthWidth = distance(mouthLeft, mouthRight);
  const mouthRatio = mouthWidth / faceWidth;
  
  // Ideal mouth width is approximately 0.5 of face width
  const idealMouthRatio = 0.5;
  const mouthScore = Math.max(0, 100 - (Math.abs(mouthRatio - idealMouthRatio) * 300));
  
  return Math.round((eyeSpacingScore * 0.4 + noseScore * 0.3 + mouthScore * 0.3));
}

/**
 * Calculate beauty ratios
 */
function calculateRatios(landmarks: Landmark[]): Array<{ name: string; value: number; idealRange: string }> {
  if (landmarks.length < 100) {
    return [
      { name: 'Face Width/Height', value: 0.65, idealRange: '0.60 - 0.70' },
      { name: 'Eye Spacing', value: 1.0, idealRange: '0.9 - 1.1' },
      { name: 'Nose Width', value: 0.28, idealRange: '0.25 - 0.30' },
      { name: 'Mouth Width', value: 0.5, idealRange: '0.45 - 0.55' }
    ];
  }
  
  const faceTop = getLandmark(landmarks, LANDMARK_INDICES.FACE_TOP);
  const faceBottom = getLandmark(landmarks, LANDMARK_INDICES.FACE_BOTTOM);
  const faceLeft = getLandmark(landmarks, LANDMARK_INDICES.FACE_LEFT);
  const faceRight = getLandmark(landmarks, LANDMARK_INDICES.FACE_RIGHT);
  
  const leftEyeCenter = calculateRegionCenter(
    landmarks,
    LANDMARK_INDICES.LEFT_EYE_REGION_START,
    LANDMARK_INDICES.LEFT_EYE_REGION_END
  );
  const rightEyeCenter = calculateRegionCenter(
    landmarks,
    LANDMARK_INDICES.RIGHT_EYE_REGION_START,
    LANDMARK_INDICES.RIGHT_EYE_REGION_END
  );
  
  const noseLeft = getLandmark(landmarks, LANDMARK_INDICES.NOSE_LEFT);
  const noseRight = getLandmark(landmarks, LANDMARK_INDICES.NOSE_RIGHT);
  const mouthLeft = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_LEFT);
  const mouthRight = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_RIGHT);
  
  // Calculate eye width from left eye region
  const leftEyePoints: Landmark[] = [];
  for (let i = LANDMARK_INDICES.LEFT_EYE_REGION_START; i <= LANDMARK_INDICES.LEFT_EYE_REGION_END && i < landmarks.length; i++) {
    if (landmarks[i]) leftEyePoints.push(landmarks[i]);
  }
  const eyeWidth = leftEyePoints.length > 1 
    ? Math.max(...leftEyePoints.map(p => p.x)) - Math.min(...leftEyePoints.map(p => p.x))
    : 0.1;
  
  const faceWidth = distance(faceLeft, faceRight);
  const faceHeight = distance(faceTop, faceBottom);
  const eyeSpacing = distance(leftEyeCenter, rightEyeCenter);
  const noseWidth = distance(noseLeft, noseRight);
  const mouthWidth = distance(mouthLeft, mouthRight);
  
  // Get additional landmarks
  const noseBridge = getLandmark(landmarks, LANDMARK_INDICES.NOSE_BRIDGE);
  const noseTip = getLandmark(landmarks, LANDMARK_INDICES.NOSE_TIP);
  const mouthTop = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_TOP);
  const mouthBottom = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_BOTTOM);

  // Calculate additional measurements
  const noseLength = distance(noseBridge, noseTip);
  const noseLengthRatio = faceHeight > 0 ? noseLength / faceHeight : 0.33;
  
  const upperLipHeight = Math.abs(mouthTop.y - (mouthTop.y + mouthBottom.y) / 2);
  const lowerLipHeight = Math.abs(mouthBottom.y - (mouthTop.y + mouthBottom.y) / 2);
  const lipRatio = lowerLipHeight > 0 ? upperLipHeight / lowerLipHeight : 0.618;
  
  const jawWidth = distance(faceLeft, faceRight);
  const jawWidthRatio = faceWidth > 0 ? jawWidth / faceWidth : 0.85;

  return [
    {
      name: 'Face Width/Height (Golden Ratio)',
      value: faceHeight > 0 ? Number((faceWidth / faceHeight).toFixed(3)) : 0.65,
      idealRange: '0.618 - 0.70 (Golden Ratio: 0.618, Modern: 0.65)'
    },
    {
      name: 'Eye Spacing Ratio',
      value: eyeWidth > 0 ? Number((eyeSpacing / eyeWidth).toFixed(3)) : 1.0,
      idealRange: '0.95 - 1.05 (Golden Ratio: 1.0, Modern: 0.95)'
    },
    {
      name: 'Nose Width Ratio',
      value: faceWidth > 0 ? Number((noseWidth / faceWidth).toFixed(3)) : 0.28,
      idealRange: '0.25 - 0.30 (Golden: 27.5%, Modern: 28%)'
    },
    {
      name: 'Nose Length Ratio',
      value: faceHeight > 0 ? Number(noseLengthRatio.toFixed(3)) : 0.33,
      idealRange: '0.30 - 0.36 (Neoclassical: 33.3%)'
    },
    {
      name: 'Mouth Width Ratio',
      value: faceWidth > 0 ? Number((mouthWidth / faceWidth).toFixed(3)) : 0.5,
      idealRange: '0.45 - 0.55 (Golden: 50%, Modern: 48%)'
    },
    {
      name: 'Lip Ratio (Upper:Lower)',
      value: Number(lipRatio.toFixed(3)),
      idealRange: '0.55 - 0.70 (Golden Ratio: 1:1.618 = 0.618)'
    },
    {
      name: 'Jaw Width Ratio',
      value: faceWidth > 0 ? Number(jawWidthRatio.toFixed(3)) : 0.85,
      idealRange: '0.80 - 0.95 (Ideal: 85-90%)'
    }
  ];
}

/**
 * Beauty Standards Definitions
 * Multiple standards used in plastic surgery and aesthetic analysis
 */
const BEAUTY_STANDARDS = {
  // Golden Ratio (Phi = 1.618)
  GOLDEN_RATIO: {
    faceWidthHeight: 0.618, // Phi inverse
    eyeSpacing: 1.0, // 1 eye width between eyes
    noseWidth: 0.275, // 27.5% of face width
    mouthWidth: 0.5, // 50% of face width
    lipRatio: 1.618, // Upper:Lower lip = 1:1.618
  },
  
  // Neoclassical Canons (Classical Greek/Roman)
  NEOCLASSICAL: {
    facialThirds: 0.333, // Equal thirds (33.3% each)
    eyeWidth: 0.1, // Eye width = 1/10 of face width
    noseLength: 0.33, // Nose length = 1/3 of face height
    chinHeight: 0.33, // Chin = 1/3 of lower third
  },
  
  // Modern Aesthetic Standards
  MODERN: {
    faceWidthHeight: 0.65, // Slightly wider than golden ratio
    eyeSpacing: 0.95, // Slightly closer than 1:1
    noseWidth: 0.28, // 28% of face width
    mouthWidth: 0.48, // 48% of face width
    jawAngle: 120, // Degrees (ideal jaw angle)
  },
  
  // Marquardt's Mask (Mathematical Beauty)
  MARQUARDT: {
    faceWidthHeight: 0.64,
    eyeSpacing: 1.0,
    noseWidth: 0.27,
    mouthWidth: 0.5,
  },
  
  // Surgical Alignment Standards
  SURGICAL: {
    symmetryTolerance: 0.02, // 2% tolerance for asymmetry
    proportionTolerance: 0.05, // 5% tolerance for proportions
    alignmentTolerance: 0.01, // 1% for feature alignment
  }
};

/**
 * Generate comprehensive surgical alignment recommendations based on multiple beauty standards
 */
function generateRecommendations(
  landmarks: Landmark[],
  ratios: Array<{ name: string; value: number; idealRange: string }>,
  symmetry: number,
  proportions: number,
  balance: number
): string[] {
  const recommendations: string[] = [];
  
  if (landmarks.length < 100) {
    return ['Insufficient landmark data for surgical analysis. Please use a clearer, front-facing image.'];
  }

  // Get key measurements
  const faceTop = getLandmark(landmarks, LANDMARK_INDICES.FACE_TOP);
  const faceBottom = getLandmark(landmarks, LANDMARK_INDICES.FACE_BOTTOM);
  const faceLeft = getLandmark(landmarks, LANDMARK_INDICES.FACE_LEFT);
  const faceRight = getLandmark(landmarks, LANDMARK_INDICES.FACE_RIGHT);
  const faceWidth = distance(faceLeft, faceRight);
  const faceHeight = distance(faceTop, faceBottom);
  const faceRatio = faceHeight > 0 ? faceWidth / faceHeight : 0;
  
  const leftEyeCenter = calculateRegionCenter(landmarks, LANDMARK_INDICES.LEFT_EYE_REGION_START, LANDMARK_INDICES.LEFT_EYE_REGION_END);
  const rightEyeCenter = calculateRegionCenter(landmarks, LANDMARK_INDICES.RIGHT_EYE_REGION_START, LANDMARK_INDICES.RIGHT_EYE_REGION_END);
  const eyeSpacing = distance(leftEyeCenter, rightEyeCenter);
  
  // Calculate eye width
  const leftEyePoints: Landmark[] = [];
  for (let i = LANDMARK_INDICES.LEFT_EYE_REGION_START; i <= LANDMARK_INDICES.LEFT_EYE_REGION_END && i < landmarks.length; i++) {
    if (landmarks[i]) leftEyePoints.push(landmarks[i]);
  }
  const eyeWidth = leftEyePoints.length > 1 
    ? Math.max(...leftEyePoints.map(p => p.x)) - Math.min(...leftEyePoints.map(p => p.x))
    : 0.1;
  const eyeSpacingRatio = eyeWidth > 0 ? eyeSpacing / eyeWidth : 1.0;
  
  const noseLeft = getLandmark(landmarks, LANDMARK_INDICES.NOSE_LEFT);
  const noseRight = getLandmark(landmarks, LANDMARK_INDICES.NOSE_RIGHT);
  const noseWidth = distance(noseLeft, noseRight);
  const noseWidthRatio = faceWidth > 0 ? noseWidth / faceWidth : 0.28;
  
  const mouthLeft = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_LEFT);
  const mouthRight = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_RIGHT);
  const mouthWidth = distance(mouthLeft, mouthRight);
  const mouthWidthRatio = faceWidth > 0 ? mouthWidth / faceWidth : 0.5;
  
  const noseTip = getLandmark(landmarks, LANDMARK_INDICES.NOSE_TIP);
  const noseBridge = getLandmark(landmarks, LANDMARK_INDICES.NOSE_BRIDGE);
  const noseLength = distance(noseBridge, noseTip);
  const noseLengthRatio = faceHeight > 0 ? noseLength / faceHeight : 0.33;
  
  const mouthTop = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_TOP);
  const mouthBottom = getLandmark(landmarks, LANDMARK_INDICES.MOUTH_BOTTOM);
  const upperLipHeight = Math.abs(mouthTop.y - (mouthTop.y + mouthBottom.y) / 2);
  const lowerLipHeight = Math.abs(mouthBottom.y - (mouthTop.y + mouthBottom.y) / 2);
  const lipRatio = lowerLipHeight > 0 ? upperLipHeight / lowerLipHeight : 0.618;

  // 1. FACE WIDTH/HEIGHT ANALYSIS (Multiple Standards)
  const goldenRatioDev = Math.abs(faceRatio - BEAUTY_STANDARDS.GOLDEN_RATIO.faceWidthHeight);
  const modernDev = Math.abs(faceRatio - BEAUTY_STANDARDS.MODERN.faceWidthHeight);
  const marquardtDev = Math.abs(faceRatio - BEAUTY_STANDARDS.MARQUARDT.faceWidthHeight);
  
    if (goldenRatioDev > BEAUTY_STANDARDS.SURGICAL.proportionTolerance || 
      modernDev > BEAUTY_STANDARDS.SURGICAL.proportionTolerance) {
      const currentPercent = (faceRatio * 100).toFixed(1);
      const idealPercent = (BEAUTY_STANDARDS.GOLDEN_RATIO.faceWidthHeight * 100).toFixed(1);
      
      if (faceRatio < BEAUTY_STANDARDS.GOLDEN_RATIO.faceWidthHeight) {
        recommendations.push(`Face width/height ratio: Currently ${currentPercent}% (Ideal: ${idealPercent}%). Face is too narrow.`);
      } else {
        recommendations.push(`Face width/height ratio: Currently ${currentPercent}% (Ideal: ${idealPercent}%). Face is too wide.`);
      }
    }

  // 2. EYE SPACING ANALYSIS
  const eyeSpacingGoldenDev = Math.abs(eyeSpacingRatio - BEAUTY_STANDARDS.GOLDEN_RATIO.eyeSpacing);
  const eyeSpacingModernDev = Math.abs(eyeSpacingRatio - BEAUTY_STANDARDS.MODERN.eyeSpacing);
  
  if (eyeSpacingGoldenDev > 0.15 || eyeSpacingModernDev > 0.15) {
    const currentPercent = (eyeSpacingRatio * 100).toFixed(0);
    if (eyeSpacingRatio < 0.85) {
      recommendations.push(`Eye spacing: Currently ${currentPercent}% of eye width (Ideal: 100%). Eyes are too close together.`);
    } else if (eyeSpacingRatio > 1.15) {
      recommendations.push(`Eye spacing: Currently ${currentPercent}% of eye width (Ideal: 100%). Eyes are too far apart.`);
    }
  }

  // 3. NOSE WIDTH ANALYSIS
  const noseGoldenDev = Math.abs(noseWidthRatio - BEAUTY_STANDARDS.GOLDEN_RATIO.noseWidth);
  const noseModernDev = Math.abs(noseWidthRatio - BEAUTY_STANDARDS.MODERN.noseWidth);
  
  if (noseGoldenDev > BEAUTY_STANDARDS.SURGICAL.proportionTolerance || 
      noseModernDev > BEAUTY_STANDARDS.SURGICAL.proportionTolerance) {
    const currentPercent = (noseWidthRatio * 100).toFixed(1);
    const idealPercent = (BEAUTY_STANDARDS.GOLDEN_RATIO.noseWidth * 100).toFixed(1);
    
    if (noseWidthRatio < 0.25) {
      recommendations.push(`Nose width: Currently ${currentPercent}% of face width (Ideal: ${idealPercent}%). Nose is too narrow.`);
    } else if (noseWidthRatio > 0.30) {
      recommendations.push(`Nose width: Currently ${currentPercent}% of face width (Ideal: ${idealPercent}%). Nose is too wide.`);
    }
  }

  // 4. MOUTH WIDTH ANALYSIS
  const mouthGoldenDev = Math.abs(mouthWidthRatio - BEAUTY_STANDARDS.GOLDEN_RATIO.mouthWidth);
  const mouthModernDev = Math.abs(mouthWidthRatio - BEAUTY_STANDARDS.MODERN.mouthWidth);
  
  if (mouthGoldenDev > BEAUTY_STANDARDS.SURGICAL.proportionTolerance || 
      mouthModernDev > BEAUTY_STANDARDS.SURGICAL.proportionTolerance) {
    const currentPercent = (mouthWidthRatio * 100).toFixed(1);
    const idealPercent = (BEAUTY_STANDARDS.GOLDEN_RATIO.mouthWidth * 100).toFixed(0);
    
    if (mouthWidthRatio < 0.45) {
      recommendations.push(`Mouth width: Currently ${currentPercent}% of face width (Ideal: ${idealPercent}%). Mouth is too narrow.`);
    } else if (mouthWidthRatio > 0.55) {
      recommendations.push(`Mouth width: Currently ${currentPercent}% of face width (Ideal: ${idealPercent}%). Mouth is too wide.`);
    }
  }

  // 5. LIP RATIO ANALYSIS (Golden Ratio: Upper:Lower = 1:1.618)
  const idealLipRatio = BEAUTY_STANDARDS.GOLDEN_RATIO.lipRatio;
  const lipRatioDev = Math.abs(lipRatio - idealLipRatio);
  
  if (lipRatioDev > 0.2) {
    const currentRatio = lipRatio.toFixed(2);
    if (lipRatio < 0.5) {
      recommendations.push(`Lip ratio: Currently ${currentRatio}:1 (Ideal: 1:1.618 Golden Ratio). Upper lip is too thin.`);
    } else if (lipRatio > 0.8) {
      recommendations.push(`Lip ratio: Currently ${currentRatio}:1 (Ideal: 1:1.618 Golden Ratio). Lower lip is too thin.`);
    }
  }

  // 6. NOSE LENGTH ANALYSIS (Neoclassical Canon: 1/3 of face height)
  const idealNoseLength = BEAUTY_STANDARDS.NEOCLASSICAL.noseLength;
  const noseLengthDev = Math.abs(noseLengthRatio - idealNoseLength);
  
  if (noseLengthDev > BEAUTY_STANDARDS.SURGICAL.proportionTolerance) {
    const currentPercent = (noseLengthRatio * 100).toFixed(1);
    if (noseLengthRatio < 0.30) {
      recommendations.push(`Nose length: Currently ${currentPercent}% of face height (Ideal: 33.3%). Nose is too short.`);
    } else if (noseLengthRatio > 0.36) {
      recommendations.push(`Nose length: Currently ${currentPercent}% of face height (Ideal: 33.3%). Nose is too long.`);
    }
  }

  // 7. FACIAL THIRDS ANALYSIS (Neoclassical Canon - Equal Thirds)
  const eyeCenterY = (leftEyeCenter.y + rightEyeCenter.y) / 2;
  const browCenter = { x: (leftEyeCenter.x + rightEyeCenter.x) / 2, y: eyeCenterY - 0.02 };
  
  const totalHeight = Math.abs(faceBottom.y - faceTop.y);
  const upperThird = Math.abs(browCenter.y - faceTop.y);
  const middleThird = Math.abs(noseTip.y - browCenter.y);
  const lowerThird = Math.abs(faceBottom.y - noseTip.y);
  
  const idealThird = totalHeight / 3;
  const upperDeviation = Math.abs(upperThird - idealThird) / idealThird;
  const middleDeviation = Math.abs(middleThird - idealThird) / idealThird;
  const lowerDeviation = Math.abs(lowerThird - idealThird) / idealThird;
  
  if (upperDeviation > 0.1) {
    const currentPercent = ((upperThird / totalHeight) * 100).toFixed(1);
    if (upperThird < idealThird) {
      recommendations.push(`Upper facial third (forehead): Currently ${currentPercent}% (Ideal: 33.3%). Forehead is too short.`);
    } else {
      recommendations.push(`Upper facial third (forehead): Currently ${currentPercent}% (Ideal: 33.3%). Forehead is too tall.`);
    }
  }
  
  if (middleDeviation > 0.1) {
    const currentPercent = ((middleThird / totalHeight) * 100).toFixed(1);
    if (middleThird < idealThird) {
      recommendations.push(`Middle facial third (eyes to nose): Currently ${currentPercent}% (Ideal: 33.3%). Middle section is too short.`);
    } else {
      recommendations.push(`Middle facial third (eyes to nose): Currently ${currentPercent}% (Ideal: 33.3%). Middle section is too tall.`);
    }
  }
  
  if (lowerDeviation > 0.1) {
    const currentPercent = ((lowerThird / totalHeight) * 100).toFixed(1);
    if (lowerThird < idealThird) {
      recommendations.push(`Lower facial third (nose to chin): Currently ${currentPercent}% (Ideal: 33.3%). Lower section is too short.`);
    } else {
      recommendations.push(`Lower facial third (nose to chin): Currently ${currentPercent}% (Ideal: 33.3%). Lower section is too tall.`);
    }
  }

  // 8. SYMMETRY ALIGNMENT ANALYSIS (Critical for Surgery)
  if (symmetry < 90) {
    const eyeVerticalDiff = Math.abs(leftEyeCenter.y - rightEyeCenter.y);
    
    if (eyeVerticalDiff > BEAUTY_STANDARDS.SURGICAL.alignmentTolerance) {
      const diffPercent = (eyeVerticalDiff * 100).toFixed(1);
      const higherEye = leftEyeCenter.y < rightEyeCenter.y ? 'left' : 'right';
      recommendations.push(`Eye asymmetry: ${diffPercent}% vertical difference detected. ${higherEye.charAt(0).toUpperCase() + higherEye.slice(1)} eye is higher than the other.`);
    }
    
    const mouthVerticalDiff = Math.abs(mouthLeft.y - mouthRight.y);
    
    if (mouthVerticalDiff > BEAUTY_STANDARDS.SURGICAL.alignmentTolerance) {
      const diffPercent = (mouthVerticalDiff * 100).toFixed(1);
      recommendations.push(`Mouth asymmetry: ${diffPercent}% vertical difference detected. Mouth is tilted.`);
    }
    
    // Nose alignment
    const faceCenterX = (faceLeft.x + faceRight.x) / 2;
    const noseTipX = noseTip.x;
    const noseDeviation = Math.abs(noseTipX - faceCenterX);
    
    if (noseDeviation > BEAUTY_STANDARDS.SURGICAL.alignmentTolerance) {
      const diffPercent = (noseDeviation * 100).toFixed(1);
      const direction = noseTipX < faceCenterX ? 'left' : 'right';
      recommendations.push(`Nose deviation: ${diffPercent}% deviation to ${direction}.`);
    }
  }

  // 9. JAW AND CHIN ANALYSIS
  const chinPoint = faceBottom;
  const jawLeft = faceLeft;
  const jawRight = faceRight;
  const jawWidth = distance(jawLeft, jawRight);
  const jawWidthRatio = faceWidth > 0 ? jawWidth / faceWidth : 0.85;
  
  // Ideal jaw width is approximately 85-90% of face width
  if (jawWidthRatio < 0.80) {
    const currentPercent = (jawWidthRatio * 100).toFixed(1);
    recommendations.push(`Jaw width: Currently ${currentPercent}% of face width (Ideal: 85-90%). Jaw is too narrow.`);
  } else if (jawWidthRatio > 0.95) {
    const currentPercent = (jawWidthRatio * 100).toFixed(1);
    recommendations.push(`Jaw width: Currently ${currentPercent}% of face width (Ideal: 85-90%). Jaw is too wide.`);
  }

  // 10. CHEEKBONE ANALYSIS
  const cheekboneHeight = Math.abs((leftEyeCenter.y + rightEyeCenter.y) / 2 - (faceTop.y + faceBottom.y) / 2);
  const idealCheekPosition = faceHeight * 0.4; // Cheekbones at 40% from top
  const cheekDeviation = Math.abs(cheekboneHeight - idealCheekPosition);
  
  if (cheekDeviation > faceHeight * 0.05) {
    if (cheekboneHeight < idealCheekPosition) {
      recommendations.push(`Cheekbone position: Cheekbones are too low.`);
    } else {
      recommendations.push(`Cheekbone position: Cheekbones are too high.`);
    }
  }

  // If no specific recommendations, show positive feedback
  if (recommendations.length === 0) {
    recommendations.push('Excellent alignment! Facial features closely match multiple beauty standards (Golden Ratio, Neoclassical, Modern).');
    recommendations.push('All proportions are within ideal ranges.');
  }

  return recommendations;
}

/**
 * Calculate overall aesthetic score (0-100)
 */
export function calculateAestheticScore(landmarks: Landmark[]): {
  score: number;
  metrics: {
    symmetry: number;
    proportions: number;
    balance: number;
  };
  ratios: Array<{ name: string; value: number; idealRange: string }>;
  recommendations: string[];
} {
  const symmetry = calculateSymmetry(landmarks);
  const proportions = calculateProportions(landmarks);
  const balance = calculateBalance(landmarks);
  
  // Weighted average for overall score
  const overallScore = Math.round(
    symmetry * 0.35 +
    proportions * 0.35 +
    balance * 0.30
  );
  
  const ratios = calculateRatios(landmarks);
  const recommendations = generateRecommendations(landmarks, ratios, symmetry, proportions, balance);
  
  return {
    score: Math.max(0, Math.min(100, overallScore)),
    metrics: {
      symmetry: Math.max(0, Math.min(100, symmetry)),
      proportions: Math.max(0, Math.min(100, proportions)),
      balance: Math.max(0, Math.min(100, balance))
    },
    ratios,
    recommendations
  };
}
