export interface BufferGeometryData {
  positions: Float32Array
  normals: Float32Array
  indices: Uint16Array
}

export interface LambertMaterialData {
  color: [number, number, number]
}

export interface AnimatedMeshSpec {
  geometry: BufferGeometryData
  material: LambertMaterialData
  position: [number, number, number]
  rotation: [number, number, number]
  rotationSpeed: [number, number, number]
}

type Vec3 = [number, number, number]

const BASE_TETRAHEDRON: readonly Vec3[] = [
  [0, 0.82, 0],
  [-0.78, -0.42, 0.44],
  [0.78, -0.42, 0.44],
  [0, -0.42, -0.88],
]

const TETRA_FACES: readonly [number, number, number][] = [
  [0, 1, 2],
  [0, 2, 3],
  [0, 3, 1],
  [1, 3, 2],
]

function mulberry32(seed: number) {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function hashIndex(index: number, salt: number) {
  let h = (index + 1) ^ salt
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b)
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b)
  return h ^ (h >>> 16)
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / len, v[1] / len, v[2] / len]
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

function subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((h * 6) % 2) - 1))
  const m = l - c * 0.5
  let r = 0
  let g = 0
  let b = 0
  const sector = (h * 6) | 0

  if (sector === 0) { r = c; g = x }
  else if (sector === 1) { r = x; g = c }
  else if (sector === 2) { g = c; b = x }
  else if (sector === 3) { g = x; b = c }
  else if (sector === 4) { r = x; b = c }
  else { r = c; b = x }

  return [r + m, g + m, b + m]
}

function fract(value: number) {
  return value - Math.floor(value)
}

function createDeformedTetrahedronVertices(index: number): Vec3[] {
  const rand = mulberry32(hashIndex(index, 0x51f15e7d))

  return BASE_TETRAHEDRON.map(([x, y, z]) => {
    const radialScale = 0.75 + rand() * 0.55
    const skewX = (rand() - 0.5) * 0.24
    const skewY = (rand() - 0.5) * 0.24
    const skewZ = (rand() - 0.5) * 0.24
    return [
      x * radialScale + skewX,
      y * radialScale + skewY,
      z * radialScale + skewZ,
    ]
  })
}

export function createUniqueTetrahedronGeometry(index: number): BufferGeometryData {
  const vertices = createDeformedTetrahedronVertices(index)
  const positions = new Float32Array(TETRA_FACES.length * 9)
  const normals = new Float32Array(TETRA_FACES.length * 9)
  const indices = new Uint16Array(TETRA_FACES.length * 3)

  let cursor = 0
  for (let faceIndex = 0; faceIndex < TETRA_FACES.length; faceIndex++) {
    const [ia, ib, ic] = TETRA_FACES[faceIndex]
    const a = vertices[ia]
    const b = vertices[ib]
    const c = vertices[ic]
    const normal = normalize(cross(subtract(b, a), subtract(c, a)))

    for (const vertex of [a, b, c]) {
      positions[cursor * 3] = vertex[0]
      positions[cursor * 3 + 1] = vertex[1]
      positions[cursor * 3 + 2] = vertex[2]

      normals[cursor * 3] = normal[0]
      normals[cursor * 3 + 1] = normal[1]
      normals[cursor * 3 + 2] = normal[2]

      indices[cursor] = cursor
      cursor++
    }
  }

  return { positions, normals, indices }
}

export function createUniqueLambertMaterial(index: number): LambertMaterialData {
  const rand = mulberry32(hashIndex(index, 0x9e3779b9))
  const hue = fract(0.13 + index * 0.6180339887498949)
  const saturation = 0.55 + rand() * 0.25
  const lightness = 0.42 + rand() * 0.18
  return {
    color: hslToRgb(hue, saturation, lightness),
  }
}

export function createUniqueAnimatedTransform(index: number) {
  const rand = mulberry32(hashIndex(index, 0x7f4a7c15))
  const spread = 50

  return {
    position: [
      (rand() - 0.5) * spread,
      (rand() - 0.5) * spread,
      (rand() - 0.5) * spread,
    ] as [number, number, number],
    rotation: [
      rand() * Math.PI * 2,
      rand() * Math.PI * 2,
      rand() * Math.PI * 2,
    ] as [number, number, number],
    rotationSpeed: [
      0.4 + rand() * 1.6,
      0.4 + rand() * 1.6,
      0.2 + rand() * 0.8,
    ] as [number, number, number],
  }
}

export function createUniqueTetrahedronSpec(index: number): AnimatedMeshSpec {
  return {
    geometry: createUniqueTetrahedronGeometry(index),
    material: createUniqueLambertMaterial(index),
    ...createUniqueAnimatedTransform(index),
  }
}
