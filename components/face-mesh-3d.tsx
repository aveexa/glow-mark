'use client'

import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Grid, Environment } from '@react-three/drei'
import * as THREE from 'three'
import { Landmark } from '@/lib/types'

interface FaceMesh3DProps {
  landmarks: Landmark[]
  className?: string
}

// MediaPipe face mesh connections (simplified - key connections for 468 landmarks)
// This creates a basic mesh structure
const FACE_MESH_CONNECTIONS = [
  // Face outline
  [10, 151, 9, 175, 18, 200, 199, 175, 9, 10],
  // Left eyebrow
  [107, 55, 65, 52, 53, 46, 107],
  // Right eyebrow
  [336, 285, 295, 282, 283, 276, 336],
  // Nose
  [4, 6, 168, 8, 98, 97, 2, 326, 327, 2, 97, 98, 8, 168, 6, 4],
  // Left eye
  [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
  // Right eye
  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362],
  // Mouth outer
  [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 61],
  // Mouth inner
  [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78],
]

function FaceMesh({ landmarks }: { landmarks: Landmark[] }) {
  const meshRef = useRef<THREE.Group>(null)

  // Create geometry from landmarks
  const { positions, indices } = useMemo(() => {
    if (landmarks.length === 0) return { positions: new Float32Array(0), indices: new Uint16Array(0) }

    const positionsArray: number[] = []
    const indicesArray: number[] = []

    // Convert normalized landmarks to 3D positions
    landmarks.forEach((landmark) => {
      // Scale and center the coordinates
      const x = (landmark.x - 0.5) * 2 // -1 to 1
      const y = -(landmark.y - 0.5) * 2 // -1 to 1 (flip Y)
      const z = (landmark.z || 0) * 0.5 // Scale depth
      positionsArray.push(x, y, z)
    })

    // Create mesh connections from predefined face mesh
    FACE_MESH_CONNECTIONS.forEach((connection) => {
      for (let i = 0; i < connection.length - 1; i++) {
        const idx1 = connection[i]
        const idx2 = connection[i + 1]
        if (idx1 < landmarks.length && idx2 < landmarks.length) {
          indicesArray.push(idx1, idx2)
        }
      }
    })

    return { 
      positions: new Float32Array(positionsArray), 
      indices: new Uint16Array(indicesArray) 
    }
  }, [landmarks])

  useFrame((state) => {
    if (meshRef.current) {
      // Subtle rotation animation
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.1
    }
  })

  if (positions.length === 0) {
    return null
  }

  return (
    <group ref={meshRef}>
      {/* Point cloud */}
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={landmarks.length}
            array={positions}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial size={0.03} color="#FFD700" sizeAttenuation={true} />
      </points>

      {/* Mesh lines */}
      {indices.length > 0 && (
        <lineSegments>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={landmarks.length}
              array={positions}
              itemSize={3}
            />
            <bufferAttribute
              attach="index"
              count={indices.length}
              array={indices}
              itemSize={1}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#FFA500" linewidth={1} />
        </lineSegments>
      )}
    </group>
  )
}

export function FaceMesh3D({ landmarks, className }: FaceMesh3DProps) {
  if (!landmarks || landmarks.length === 0) {
    return (
      <div className={`flex items-center justify-center bg-black/5 rounded-2xl ${className || ''}`}>
        <p className="text-muted-foreground">No landmark data available for 3D visualization</p>
      </div>
    )
  }

  return (
    <div className={`relative bg-black/5 rounded-2xl overflow-hidden ${className || ''}`} style={{ height: '600px' }}>
      <Canvas
        camera={{ position: [0, 0, 2], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        className="w-full h-full"
      >
        <PerspectiveCamera makeDefault position={[0, 0, 2]} fov={50} />
        
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={1} />
        <directionalLight position={[-5, -5, -5]} intensity={0.5} />
        <pointLight position={[0, 0, 5]} intensity={0.5} />

        {/* Grid helper */}
        <Grid args={[10, 10]} cellColor="#333333" sectionColor="#444444" />

        {/* Face mesh */}
        <FaceMesh landmarks={landmarks} />

        {/* Controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={1}
          maxDistance={5}
          autoRotate={false}
        />
      </Canvas>
      
      {/* Instructions overlay */}
      <div className="absolute bottom-4 left-4 right-4 bg-black/60 backdrop-blur-md rounded-lg p-3 text-white text-xs">
        <p className="font-semibold mb-1">3D Face Mesh Visualization</p>
        <p className="text-white/80">Drag to rotate • Scroll to zoom • Right-click to pan</p>
      </div>
    </div>
  )
}
