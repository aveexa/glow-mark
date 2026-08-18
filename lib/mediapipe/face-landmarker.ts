import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

let faceLandmarker: FaceLandmarker | null = null;
let isInitialized = false;

/**
 * Initialize MediaPipe Face Landmarker
 */
export async function initializeFaceLandmarker(): Promise<FaceLandmarker> {
  if (faceLandmarker && isInitialized) {
    return faceLandmarker;
  }

  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
        delegate: "GPU"
      },
      outputFaceBlendshapes: false,
      runningMode: "IMAGE",
      numFaces: 1, // Only detect one face
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    isInitialized = true;
    return faceLandmarker;
  } catch (error) {
    console.error('Error initializing Face Landmarker:', error);
    throw error;
  }
}

/**
 * Detect face landmarks from an image
 */
export async function detectFaceLandmarks(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<{
  landmarks: Array<{ x: number; y: number }>;
  faceDetected: boolean;
  faceCount: number;
}> {
  try {
    const landmarker = await initializeFaceLandmarker();
    const results = landmarker.detect(imageElement);

    if (!results.faceLandmarks || results.faceLandmarks.length === 0) {
      return {
        landmarks: [],
        faceDetected: false,
        faceCount: 0
      };
    }

    if (results.faceLandmarks.length > 1) {
      return {
        landmarks: [],
        faceDetected: false,
        faceCount: results.faceLandmarks.length
      };
    }

    // Convert MediaPipe landmarks to our format (normalized 0-1)
    // MediaPipe returns landmarks as {x, y, z} where x and y are normalized 0-1, z is depth
    const landmarks = results.faceLandmarks[0].map((landmark: any) => ({
      x: landmark.x || 0,
      y: landmark.y || 0,
      z: landmark.z || 0 // Depth coordinate for 3D visualization
    }));

    return {
      landmarks,
      faceDetected: true,
      faceCount: 1
    };
  } catch (error) {
    console.error('Error detecting face landmarks:', error);
    throw error;
  }
}

/**
 * Convert image file to HTMLImageElement
 */
export function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    
    img.onerror = (error) => {
      URL.revokeObjectURL(url);
      reject(error);
    };
    
    img.src = url;
  });
}
