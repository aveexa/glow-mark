# GlowMark - Facial Analysis Frontend

A Next.js application for facial analysis and beauty scoring with a focus on privacy and user experience.

## Features

- **Image Upload**: Drag & drop or file picker with client-side validation
- **Real-time Preview**: Instant preview of selected images
- **Processing Stepper**: Visual progress indicator during analysis
- **Results Dashboard**: Comprehensive analysis results with overlay visualization
- **Error Handling**: Comprehensive error states with user guidance
- **Privacy First**: No permanent image storage - all processing in memory

## Tech Stack

- **Next.js 14** (App Router)
- **TypeScript**
- **TailwindCSS**
- **shadcn/ui** components
- **react-hook-form** + **zod** for validation
- **Zustand** for state management

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

## Project Structure

```
glow-mark/
├── app/
│   ├── api/
│   │   └── analyze/
│   │       └── route.ts          # Mock API endpoint
│   ├── analyze/
│   │   └── page.tsx              # Main analysis page
│   ├── privacy/
│   │   └── page.tsx              # Privacy policy page
│   ├── layout.tsx                # Root layout
│   ├── page.tsx                  # Landing page
│   └── globals.css               # Global styles
├── components/
│   ├── ui/                       # shadcn/ui components
│   ├── upload-dropzone.tsx       # File upload component
│   ├── preview-panel.tsx         # Image preview component
│   ├── processing-stepper.tsx    # Processing progress
│   ├── results-dashboard.tsx     # Results display
│   └── error-state.tsx           # Error display
├── lib/
│   ├── constants.ts              # App constants
│   ├── schemas.ts                # Zod validation schemas
│   ├── types.ts                  # TypeScript types
│   └── utils.ts                  # Utility functions
├── store/
│   └── analysis-store.ts         # Zustand state store
└── hooks/
    └── use-toast.ts              # Toast notification hook
```

## API Mock - Forcing Error States

The mock API endpoint at `/api/analyze` supports forcing specific error states for testing purposes.

### Usage

Add a `force` query parameter to the API request with one of the following error codes:

```
POST /api/analyze?force=ERROR_CODE
```

### Available Error Codes

- `INVALID_FILE_TYPE` - File type is not JPEG, PNG, or WebP
- `FILE_TOO_LARGE` - File size exceeds 5MB
- `CORRUPT_FILE` - Image file appears corrupted
- `NO_FACE_DETECTED` - No face detected in the image
- `MULTIPLE_FACES_DETECTED` - Multiple faces detected
- `FACE_TOO_ANGLED_OR_SMALL` - Face is too angled or too small
- `TIMEOUT` - Processing timed out
- `UNKNOWN_ERROR` - Unexpected error occurred

### Example

To test the "No Face Detected" error:

```typescript
const response = await fetch('/api/analyze?force=NO_FACE_DETECTED', {
  method: 'POST',
  body: formData,
})
```

### Normal Operation

Without the `force` parameter, the API will:
- Process the image (1-3 second delay)
- Return a mock analysis result with:
  - Score (0-100)
  - Metrics (symmetry, proportions, balance)
  - Landmarks (normalized coordinates)
  - Ratios and recommendations
  - Research notes

**Note**: The API randomly returns errors 10% of the time for testing purposes (unless a specific error is forced).

## Privacy & Data Handling

- **No Permanent Storage**: Images are never written to disk or stored in databases
- **In-Memory Processing**: All image processing happens in temporary memory
- **Automatic Cleanup**: Object URLs are revoked when:
  - User navigates away
  - User clicks "Delete Now"
  - User refreshes the page
  - Browser tab is closed
- **No Tracking**: No analytics or third-party tracking

## Component Details

### UploadDropzone
- Drag & drop support
- File picker fallback
- Client-side validation (type, size)
- Inline error display

### PreviewPanel
- Image preview with aspect ratio preservation
- Actions: Analyze, Change image, Remove
- Disabled states during processing

### ProcessingStepper
- Step-by-step progress indicator
- Animated progress bar
- Cancel functionality

### ResultsDashboard
- Two-column layout (image + metrics)
- Canvas overlay for landmarks (toggleable)
- Score gauge visualization
- Tabbed content (Insights, Ratios, Recommendations, Notes)
- Action buttons with cleanup

### ErrorState
- User-friendly error messages
- Retry and cancel actions
- Accessible error display

## State Management

The application uses Zustand for session state management:

- `selectedFile`: Currently selected file
- `previewUrl`: Object URL for preview
- `analysisStatus`: Current analysis state
- `progressStep`: Current processing step
- `result`: Analysis results
- `error`: Error code if any

All state is cleared on:
- Route change
- Page refresh
- Explicit "Delete Now" action
- Component unmount

## Development Notes

- All components are client-side (`"use client"`)
- Form validation uses react-hook-form with zod
- Toast notifications for user feedback
- Responsive design with TailwindCSS
- Accessibility features (ARIA labels, keyboard navigation)

## License

Private project - All rights reserved.
