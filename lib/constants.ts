export const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB in bytes
export const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export const PROCESSING_STEPS = [
  'Validating',
  'Face detection',
  'Landmark detection',
  'Calculations',
  'Score & Recommendations'
] as const;

export type ProcessingStep = typeof PROCESSING_STEPS[number];

export const ERROR_CODES = {
  INVALID_FILE_TYPE: 'INVALID_FILE_TYPE',
  FILE_TOO_LARGE: 'FILE_TOO_LARGE',
  CORRUPT_FILE: 'CORRUPT_FILE',
  NO_FACE_DETECTED: 'NO_FACE_DETECTED',
  MULTIPLE_FACES_DETECTED: 'MULTIPLE_FACES_DETECTED',
  FACE_TOO_ANGLED_OR_SMALL: 'FACE_TOO_ANGLED_OR_SMALL',
  TIMEOUT: 'TIMEOUT',
  UNKNOWN_ERROR: 'UNKNOWN_ERROR',
} as const;

export type ErrorCode = typeof ERROR_CODES[keyof typeof ERROR_CODES];

export const ERROR_MESSAGES: Record<ErrorCode, string> = {
  [ERROR_CODES.INVALID_FILE_TYPE]: 'Please upload a JPEG, PNG, or WebP image.',
  [ERROR_CODES.FILE_TOO_LARGE]: 'File size must be less than 5MB.',
  [ERROR_CODES.CORRUPT_FILE]: 'The image file appears to be corrupted. Please try another image.',
  [ERROR_CODES.NO_FACE_DETECTED]: 'No face detected in the image. Please upload a clear front-facing photo.',
  [ERROR_CODES.MULTIPLE_FACES_DETECTED]: 'Multiple faces detected. Please upload an image with a single face.',
  [ERROR_CODES.FACE_TOO_ANGLED_OR_SMALL]: 'Face is too angled or too small. Please upload a clear front-facing photo.',
  [ERROR_CODES.TIMEOUT]: 'Processing timed out. Please try again.',
  [ERROR_CODES.UNKNOWN_ERROR]: 'An unexpected error occurred. Please try again.',
};
