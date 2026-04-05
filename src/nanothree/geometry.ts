// Geometry classes for nanothree - Three.js-compatible API

export class Float32BufferAttribute {
  readonly array: Float32Array
  readonly itemSize: number

  constructor(array: ArrayLike<number>, itemSize: number) {
    this.array = array instanceof Float32Array ? array : new Float32Array(array)
    this.itemSize = itemSize
  }
}

export class BufferGeometry {
  positions: Float32Array | null = null
  normals: Float32Array | null = null
  indices: Uint16Array | Uint32Array | null = null

  // GPU resources (lazily created by renderer)
  _vertexBuffer: GPUBuffer | null = null
  _indexBuffer: GPUBuffer | null = null
  _indexCount = 0
  _indexFormat: GPUIndexFormat = 'uint16'
  _vertexCount = 0
  _gpuDirty = true
  _device: GPUDevice | null = null

  // Wireframe index buffer (lazily generated from triangle indices)
  _wireframeIndexBuffer: GPUBuffer | null = null
  _wireframeIndexCount = 0
  _wireframeIndexFormat: GPUIndexFormat = 'uint16'
  _wireframeDirty = true

  setAttribute(name: string, attribute: Float32BufferAttribute) {
    if (name === 'position') {
      this.positions = attribute.array
    } else if (name === 'normal') {
      this.normals = attribute.array
    }
    this._gpuDirty = true
    this._wireframeDirty = true
    return this
  }

  setIndex(indices: ArrayLike<number>) {
    if (indices instanceof Uint16Array) {
      this.indices = indices
    } else if (indices instanceof Uint32Array) {
      this.indices = indices
    } else {
      this.indices = new Uint16Array(indices)
    }
    this._gpuDirty = true
    this._wireframeDirty = true
    return this
  }

  _ensureGPU(device: GPUDevice) {
    if (!this._gpuDirty && this._device === device) return
    this._device = device

    const positions = this.positions!
    const normals = this.normals
    const vertexCount = positions.length / 3
    this._vertexCount = vertexCount

    // Interleave position + normal (normals default to 0 if absent)
    const interleaved = new Float32Array(vertexCount * 6)
    for (let i = 0; i < vertexCount; i++) {
      const i3 = i * 3
      const i6 = i * 6
      interleaved[i6] = positions[i3]
      interleaved[i6 + 1] = positions[i3 + 1]
      interleaved[i6 + 2] = positions[i3 + 2]
      if (normals) {
        interleaved[i6 + 3] = normals[i3]
        interleaved[i6 + 4] = normals[i3 + 1]
        interleaved[i6 + 5] = normals[i3 + 2]
      }
    }

    if (this._vertexBuffer) this._vertexBuffer.destroy()
    this._vertexBuffer = device.createBuffer({
      size: interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this._vertexBuffer, 0, interleaved)

    if (this.indices) {
      const idx = this.indices
      this._indexCount = idx.length
      this._indexFormat = idx instanceof Uint32Array ? 'uint32' : 'uint16'
      if (this._indexBuffer) this._indexBuffer.destroy()
      this._indexBuffer = device.createBuffer({
        size: idx.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(this._indexBuffer, 0, idx)
    } else {
      this._indexCount = 0
      if (this._indexBuffer) {
        this._indexBuffer.destroy()
        this._indexBuffer = null
      }
    }

    this._gpuDirty = false
  }

  _ensureWireframeGPU(device: GPUDevice) {
    this._ensureGPU(device)
    if (!this._wireframeDirty && this._device === device) return
    if (!this.indices) return

    const triIndices = this.indices
    const triCount = triIndices.length / 3
    const use32 = triIndices instanceof Uint32Array
    this._wireframeIndexFormat = use32 ? 'uint32' : 'uint16'
    const wireIndices = use32 ? new Uint32Array(triCount * 6) : new Uint16Array(triCount * 6)

    for (let i = 0; i < triCount; i++) {
      const i3 = i * 3
      const a = triIndices[i3], b = triIndices[i3 + 1], c = triIndices[i3 + 2]
      const i6 = i * 6
      wireIndices[i6] = a;     wireIndices[i6 + 1] = b
      wireIndices[i6 + 2] = b; wireIndices[i6 + 3] = c
      wireIndices[i6 + 4] = c; wireIndices[i6 + 5] = a
    }

    this._wireframeIndexCount = wireIndices.length
    if (this._wireframeIndexBuffer) this._wireframeIndexBuffer.destroy()
    this._wireframeIndexBuffer = device.createBuffer({
      size: wireIndices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this._wireframeIndexBuffer, 0, wireIndices)
    this._wireframeDirty = false
  }

  dispose() {
    this._vertexBuffer?.destroy()
    this._indexBuffer?.destroy()
    this._wireframeIndexBuffer?.destroy()
    this._vertexBuffer = null
    this._indexBuffer = null
    this._wireframeIndexBuffer = null
    this._device = null
    this._gpuDirty = true
    this._wireframeDirty = true
  }
}

export class BoxGeometry extends BufferGeometry {
  constructor(width = 1, height = 1, depth = 1) {
    super()
    const w = width / 2, h = height / 2, d = depth / 2

    this.positions = new Float32Array([
       w, -h, -d,   w,  h, -d,   w,  h,  d,   w, -h,  d,
      -w, -h,  d,  -w,  h,  d,  -w,  h, -d,  -w, -h, -d,
      -w,  h,  d,   w,  h,  d,   w,  h, -d,  -w,  h, -d,
      -w, -h, -d,   w, -h, -d,   w, -h,  d,  -w, -h,  d,
      -w, -h,  d,   w, -h,  d,   w,  h,  d,  -w,  h,  d,
       w, -h, -d,  -w, -h, -d,  -w,  h, -d,   w,  h, -d,
    ])

    this.normals = new Float32Array([
       1, 0, 0,   1, 0, 0,   1, 0, 0,   1, 0, 0,
      -1, 0, 0,  -1, 0, 0,  -1, 0, 0,  -1, 0, 0,
       0, 1, 0,   0, 1, 0,   0, 1, 0,   0, 1, 0,
       0,-1, 0,   0,-1, 0,   0,-1, 0,   0,-1, 0,
       0, 0, 1,   0, 0, 1,   0, 0, 1,   0, 0, 1,
       0, 0,-1,   0, 0,-1,   0, 0,-1,   0, 0,-1,
    ])

    this.indices = new Uint16Array([
      0, 1, 2, 0, 2, 3,
      4, 5, 6, 4, 6, 7,
      8, 9, 10, 8, 10, 11,
      12, 13, 14, 12, 14, 15,
      16, 17, 18, 16, 18, 19,
      20, 21, 22, 20, 22, 23,
    ])
  }
}

export class TetrahedronGeometry extends BufferGeometry {
  constructor(radius = 1) {
    super()
    const a = radius
    const vertices: [number, number, number][] = [
      [ a,  a,  a],
      [-a, -a,  a],
      [-a,  a, -a],
      [ a, -a, -a],
    ]
    const faces: [number, number, number][] = [
      [0, 1, 2],
      [0, 3, 1],
      [0, 2, 3],
      [1, 3, 2],
    ]

    const positions = new Float32Array(faces.length * 9)
    const normals = new Float32Array(faces.length * 9)
    const indices = new Uint16Array(faces.length * 3)

    let cursor = 0
    for (const [ia, ib, ic] of faces) {
      const va = vertices[ia], vb = vertices[ib], vc = vertices[ic]
      const ux = vb[0] - va[0], uy = vb[1] - va[1], uz = vb[2] - va[2]
      const vx = vc[0] - va[0], vy = vc[1] - va[1], vz = vc[2] - va[2]
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1
      nx /= len; ny /= len; nz /= len

      for (const v of [va, vb, vc]) {
        positions[cursor * 3] = v[0]
        positions[cursor * 3 + 1] = v[1]
        positions[cursor * 3 + 2] = v[2]
        normals[cursor * 3] = nx
        normals[cursor * 3 + 1] = ny
        normals[cursor * 3 + 2] = nz
        indices[cursor] = cursor
        cursor++
      }
    }

    this.positions = positions
    this.normals = normals
    this.indices = indices
  }
}
