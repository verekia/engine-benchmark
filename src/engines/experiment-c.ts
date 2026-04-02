import type { EngineAdapter, BackendType, UseCase } from '../types'

// ============================================================================
// WGSL Shader
// ============================================================================

const SHADER = /* wgsl */ `
struct Scene {
  viewProj: mat4x4f,
  lightDir: vec4f,
  ambient: vec4f,
  lightColor: vec4f,
}

struct NodeData {
  col0: vec4f,
  col1: vec4f,
  col2: vec4f,
  col3: vec4f,
  color: vec4f,
}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> nodes: array<NodeData>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) normal: vec3f,
  @location(1) color: vec3f,
}

@vertex fn vs(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @builtin(instance_index) idx: u32,
) -> VOut {
  let n = nodes[idx];
  let model = mat4x4f(n.col0, n.col1, n.col2, n.col3);
  var out: VOut;
  out.pos = scene.viewProj * model * vec4f(position, 1.0);
  out.normal = normalize((model * vec4f(normal, 0.0)).xyz);
  out.color = n.color.rgb;
  return out;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let diff = max(dot(n, scene.lightDir.xyz), 0.0);
  let color = in.color * (scene.ambient.rgb + scene.lightColor.rgb * diff);
  return vec4f(color, 1.0);
}
`

// ============================================================================
// Math (column-major mat4)
// ============================================================================

function mat4Perspective(out: Float32Array, fov: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fov / 2)
  const nf = 1 / (near - far)
  out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0
  out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0
  out[8] = 0; out[9] = 0; out[10] = far * nf; out[11] = -1
  out[12] = 0; out[13] = 0; out[14] = near * far * nf; out[15] = 0
}

function mat4LookAt(
  out: Float32Array,
  ex: number, ey: number, ez: number,
  tx: number, ty: number, tz: number,
  ux: number, uy: number, uz: number,
) {
  let fx = tx - ex, fy = ty - ey, fz = tz - ez
  let len = 1 / Math.sqrt(fx * fx + fy * fy + fz * fz)
  fx *= len; fy *= len; fz *= len
  let rx = fy * uz - fz * uy, ry = fz * ux - fx * uz, rz = fx * uy - fy * ux
  len = 1 / Math.sqrt(rx * rx + ry * ry + rz * rz)
  rx *= len; ry *= len; rz *= len
  const sx = ry * fz - rz * fy, sy = rz * fx - rx * fz, sz = rx * fy - ry * fx
  out[0] = rx;  out[1] = sx;  out[2] = -fx; out[3] = 0
  out[4] = ry;  out[5] = sy;  out[6] = -fy; out[7] = 0
  out[8] = rz;  out[9] = sz;  out[10] = -fz; out[11] = 0
  out[12] = -(rx * ex + ry * ey + rz * ez)
  out[13] = -(sx * ex + sy * ey + sz * ez)
  out[14] = fx * ex + fy * ey + fz * ez
  out[15] = 1
}

function mat4Mul(out: Float32Array, a: Float32Array, b: Float32Array) {
  for (let i = 0; i < 4; i++) {
    const ai0 = a[i], ai1 = a[i + 4], ai2 = a[i + 8], ai3 = a[i + 12]
    out[i]      = ai0 * b[0]  + ai1 * b[1]  + ai2 * b[2]  + ai3 * b[3]
    out[i + 4]  = ai0 * b[4]  + ai1 * b[5]  + ai2 * b[6]  + ai3 * b[7]
    out[i + 8]  = ai0 * b[8]  + ai1 * b[9]  + ai2 * b[10] + ai3 * b[11]
    out[i + 12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15]
  }
}

// ============================================================================
// Mesh
// ============================================================================

export class Mesh {
  readonly vertexBuffer: GPUBuffer
  readonly indexBuffer: GPUBuffer
  readonly indexCount: number

  constructor(device: GPUDevice, vertices: Float32Array, indices: Uint16Array) {
    this.indexCount = indices.length
    this.vertexBuffer = device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.vertexBuffer, 0, vertices)
    this.indexBuffer = device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.indexBuffer, 0, indices)
  }

  destroy() {
    this.vertexBuffer.destroy()
    this.indexBuffer.destroy()
  }
}

function createBoxGeometry(): { vertices: Float32Array; indices: Uint16Array } {
  // prettier-ignore
  const vertices = new Float32Array([
    // +X
     .5, -.5, -.5,  1, 0, 0,   .5,  .5, -.5,  1, 0, 0,   .5,  .5,  .5,  1, 0, 0,   .5, -.5,  .5,  1, 0, 0,
    // -X
    -.5, -.5,  .5, -1, 0, 0,  -.5,  .5,  .5, -1, 0, 0,  -.5,  .5, -.5, -1, 0, 0,  -.5, -.5, -.5, -1, 0, 0,
    // +Y
    -.5,  .5,  .5,  0, 1, 0,   .5,  .5,  .5,  0, 1, 0,   .5,  .5, -.5,  0, 1, 0,  -.5,  .5, -.5,  0, 1, 0,
    // -Y
    -.5, -.5, -.5,  0,-1, 0,   .5, -.5, -.5,  0,-1, 0,   .5, -.5,  .5,  0,-1, 0,  -.5, -.5,  .5,  0,-1, 0,
    // +Z
    -.5, -.5,  .5,  0, 0, 1,   .5, -.5,  .5,  0, 0, 1,   .5,  .5,  .5,  0, 0, 1,  -.5,  .5,  .5,  0, 0, 1,
    // -Z
     .5, -.5, -.5,  0, 0,-1,  -.5, -.5, -.5,  0, 0,-1,  -.5,  .5, -.5,  0, 0,-1,   .5,  .5, -.5,  0, 0,-1,
  ])
  // prettier-ignore
  const indices = new Uint16Array([
    0,1,2, 0,2,3,  4,5,6, 4,6,7,  8,9,10, 8,10,11,
    12,13,14, 12,14,15,  16,17,18, 16,18,19,  20,21,22, 20,22,23,
  ])
  return { vertices, indices }
}

// ============================================================================
// Node
// ============================================================================

const FLOATS_PER_NODE = 20 // mat4(16) + color vec4(4)

export class Node {
  x = 0; y = 0; z = 0
  rx = 0; ry = 0; rz = 0
  sx = 1; sy = 1; sz = 1
  mesh: Mesh | null = null

  /** @internal */ _buf!: Float32Array
  /** @internal */ _offset = 0
  /** @internal */ _index = -1

  setColor(r: number, g: number, b: number, a = 1) {
    const o = this._offset
    this._buf[o + 16] = r; this._buf[o + 17] = g; this._buf[o + 18] = b; this._buf[o + 19] = a
  }

  updateTransform() {
    const b = this._buf, o = this._offset
    const cosX = Math.cos(this.rx), sinX = Math.sin(this.rx)
    const cosY = Math.cos(this.ry), sinY = Math.sin(this.ry)
    const cosZ = Math.cos(this.rz), sinZ = Math.sin(this.rz)
    const { sx, sy, sz } = this

    // TRS = Translate * RotZ * RotY * RotX * Scale (column-major)
    b[o     ] = (cosZ * cosY) * sx
    b[o +  1] = (sinZ * cosY) * sx
    b[o +  2] = (-sinY) * sx
    b[o +  3] = 0
    b[o +  4] = (cosZ * sinY * sinX - sinZ * cosX) * sy
    b[o +  5] = (sinZ * sinY * sinX + cosZ * cosX) * sy
    b[o +  6] = (cosY * sinX) * sy
    b[o +  7] = 0
    b[o +  8] = (cosZ * sinY * cosX + sinZ * sinX) * sz
    b[o +  9] = (sinZ * sinY * cosX - cosZ * sinX) * sz
    b[o + 10] = (cosY * cosX) * sz
    b[o + 11] = 0
    b[o + 12] = this.x
    b[o + 13] = this.y
    b[o + 14] = this.z
    b[o + 15] = 1
  }
}

// ============================================================================
// Camera
// ============================================================================

export class Camera {
  readonly position = new Float32Array([0, 40, 72])
  readonly target = new Float32Array(3)
  fov = Math.PI / 3
  near = 0.1
  far = 1000

  private proj = new Float32Array(16)
  private view = new Float32Array(16)
  readonly viewProjection = new Float32Array(16)

  update(aspect: number) {
    mat4Perspective(this.proj, this.fov, aspect, this.near, this.far)
    mat4LookAt(
      this.view,
      this.position[0], this.position[1], this.position[2],
      this.target[0], this.target[1], this.target[2],
      0, 1, 0,
    )
    mat4Mul(this.viewProjection, this.proj, this.view)
  }
}

// ============================================================================
// Renderer
// ============================================================================

const INITIAL_CAPACITY = 1024

export class Renderer {
  readonly device: GPUDevice
  private context: GPUCanvasContext
  private canvas: HTMLCanvasElement

  private pipeline!: GPURenderPipeline
  private sceneBuffer!: GPUBuffer
  private nodeBuffer!: GPUBuffer
  private bindGroupLayout!: GPUBindGroupLayout
  private bindGroup!: GPUBindGroup
  private depthTexture!: GPUTexture
  private depthView!: GPUTextureView

  // Scene data: viewProj(16) + lightDir(4) + ambient(4) + lightColor(4)
  private sceneData = new Float32Array(28)
  private nodeData: Float32Array
  private capacity: number

  readonly nodes: Node[] = []

  // Pre-allocated render pass descriptor (mutated each frame to avoid allocation)
  private colorAtt: GPURenderPassColorAttachment
  private depthAtt: GPURenderPassDepthStencilAttachment
  private passDesc: GPURenderPassDescriptor
  private depthW = 0
  private depthH = 0

  private constructor(device: GPUDevice, context: GPUCanvasContext, canvas: HTMLCanvasElement) {
    this.device = device
    this.context = context
    this.canvas = canvas
    this.capacity = INITIAL_CAPACITY
    this.nodeData = new Float32Array(INITIAL_CAPACITY * FLOATS_PER_NODE)

    this.colorAtt = {
      view: undefined as unknown as GPUTextureView,
      clearValue: { r: 0.05, g: 0.05, b: 0.07, a: 1 },
      loadOp: 'clear' as const,
      storeOp: 'store' as const,
    }
    this.depthAtt = {
      view: undefined as unknown as GPUTextureView,
      depthClearValue: 1,
      depthLoadOp: 'clear' as const,
      depthStoreOp: 'store' as const,
    }
    this.passDesc = {
      colorAttachments: [this.colorAtt],
      depthStencilAttachment: this.depthAtt,
    }

    // Default lighting
    this.setLightDirection(0.5, 1.0, 0.3)
    this.setAmbientColor(0.15, 0.15, 0.18)
    this.setLightColor(1.0, 0.95, 0.9)
  }

  static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
    if (!navigator.gpu) throw new Error('WebGPU not supported')
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('No WebGPU adapter found')
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    }) as GPUDevice

    const context = canvas.getContext('webgpu')!
    const format = navigator.gpu.getPreferredCanvasFormat()

    const dpr = devicePixelRatio
    canvas.width = canvas.clientWidth * dpr
    canvas.height = canvas.clientHeight * dpr

    context.configure({ device, format, alphaMode: 'premultiplied' })

    const renderer = new Renderer(device, context, canvas)
    renderer.initGPU(format)
    return renderer
  }

  private initGPU(format: GPUTextureFormat) {
    const { device } = this

    this.sceneBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.nodeBuffer = device.createBuffer({
      size: this.capacity * FLOATS_PER_NODE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    })

    this.rebuildBindGroup()

    const shader = device.createShaderModule({ code: SHADER })
    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: {
        module: shader,
        entryPoint: 'vs',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: shader, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    })

    this.ensureDepthTexture()
  }

  private rebuildBindGroup() {
    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.sceneBuffer } },
        { binding: 1, resource: { buffer: this.nodeBuffer } },
      ],
    })
  }

  private ensureDepthTexture() {
    const w = this.canvas.width, h = this.canvas.height
    if (w === this.depthW && h === this.depthH) return
    this.depthW = w; this.depthH = h
    if (this.depthTexture) this.depthTexture.destroy()
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.depthView = this.depthTexture.createView()
  }

  setLightDirection(x: number, y: number, z: number) {
    const len = 1 / Math.sqrt(x * x + y * y + z * z)
    this.sceneData[16] = x * len; this.sceneData[17] = y * len; this.sceneData[18] = z * len
  }

  setAmbientColor(r: number, g: number, b: number) {
    this.sceneData[20] = r; this.sceneData[21] = g; this.sceneData[22] = b
  }

  setLightColor(r: number, g: number, b: number) {
    this.sceneData[24] = r; this.sceneData[25] = g; this.sceneData[26] = b
  }

  createNode(): Node {
    const idx = this.nodes.length
    if (idx >= this.capacity) this.grow(this.capacity * 2)

    const node = new Node()
    node._buf = this.nodeData
    node._offset = idx * FLOATS_PER_NODE
    node._index = idx
    this.nodes.push(node)
    return node
  }

  removeNode(node: Node) {
    const idx = node._index
    if (idx === -1) return
    const last = this.nodes.length - 1

    if (idx !== last) {
      const lastNode = this.nodes[last]
      this.nodes[idx] = lastNode
      lastNode._index = idx
      lastNode._offset = idx * FLOATS_PER_NODE
      this.nodeData.copyWithin(idx * FLOATS_PER_NODE, last * FLOATS_PER_NODE, (last + 1) * FLOATS_PER_NODE)
    }

    node._index = -1
    this.nodes.pop()
  }

  private grow(newCapacity: number) {
    this.capacity = newCapacity
    const newData = new Float32Array(newCapacity * FLOATS_PER_NODE)
    newData.set(this.nodeData)
    this.nodeData = newData
    for (const n of this.nodes) n._buf = newData

    this.nodeBuffer.destroy()
    this.nodeBuffer = this.device.createBuffer({
      size: newCapacity * FLOATS_PER_NODE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.rebuildBindGroup()
  }

  handleResize() {
    const dpr = devicePixelRatio
    const w = (this.canvas.clientWidth * dpr) | 0
    const h = (this.canvas.clientHeight * dpr) | 0
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h
      this.ensureDepthTexture()
    }
  }

  render(camera: Camera) {
    const n = this.nodes.length
    if (n === 0) return

    this.handleResize()

    // Upload scene uniform (viewProj + lighting)
    camera.update(this.canvas.width / this.canvas.height)
    this.sceneData.set(camera.viewProjection, 0)
    this.device.queue.writeBuffer(this.sceneBuffer, 0, this.sceneData)

    // Upload node storage buffer
    this.device.queue.writeBuffer(this.nodeBuffer, 0, this.nodeData.buffer, 0, n * FLOATS_PER_NODE * 4)

    // Encode render pass
    this.colorAtt.view = this.context.getCurrentTexture().createView()
    this.depthAtt.view = this.depthView

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(this.passDesc)

    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)

    // Hot path: one drawIndexed per node, minimal state changes
    let curMesh: Mesh | null = null
    for (let i = 0; i < n; i++) {
      const mesh = this.nodes[i].mesh
      if (!mesh) continue
      if (mesh !== curMesh) {
        curMesh = mesh
        pass.setVertexBuffer(0, mesh.vertexBuffer)
        pass.setIndexBuffer(mesh.indexBuffer, 'uint16')
      }
      pass.drawIndexed(mesh.indexCount, 1, 0, 0, i)
    }

    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  dispose() {
    this.sceneBuffer?.destroy()
    this.nodeBuffer?.destroy()
    this.depthTexture?.destroy()
    this.device?.destroy()
    this.nodes.length = 0
  }
}

// ============================================================================
// Helpers
// ============================================================================

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((h * 6) % 2) - 1))
  const m = l - c * 0.5
  let r = 0, g = 0, b = 0
  const sec = (h * 6) | 0
  if (sec === 0) { r = c; g = x }
  else if (sec === 1) { r = x; g = c }
  else if (sec === 2) { g = c; b = x }
  else if (sec === 3) { g = x; b = c }
  else if (sec === 4) { r = x; b = c }
  else { r = c; b = x }
  return [r + m, g + m, b + m]
}

// ============================================================================
// Benchmark Adapter
// ============================================================================

interface CubeState {
  node: Node
  rxSpeed: number
  rySpeed: number
}

export class ExperimentCAdapter implements EngineAdapter {
  private renderer!: Renderer
  private camera = new Camera()
  private cubes: CubeState[] = []
  private boxMesh!: Mesh
  private canvas!: HTMLCanvasElement
  private useCase: UseCase = 'boxes'

  // Orbit camera state
  private camTheta = 0
  private camPhi = Math.acos(40 / 72)
  private camDist = 72
  private dragging = false
  private lastMX = 0
  private lastMY = 0

  async init(canvas: HTMLCanvasElement, backend: BackendType, useCase: UseCase) {
    if (backend !== 'webgpu') throw new Error('Experiment C supports WebGPU only')
    this.canvas = canvas
    this.useCase = useCase

    this.renderer = await Renderer.create(canvas)

    const { vertices, indices } = createBoxGeometry()
    this.boxMesh = new Mesh(this.renderer.device, vertices, indices)

    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerup', this.onPointerUp)
    canvas.addEventListener('wheel', this.onWheel, { passive: true })
  }

  private onPointerDown = (e: PointerEvent) => {
    this.dragging = true
    this.lastMX = e.clientX; this.lastMY = e.clientY
    this.canvas.setPointerCapture(e.pointerId)
  }

  private onPointerMove = (e: PointerEvent) => {
    if (!this.dragging) return
    this.camTheta -= (e.clientX - this.lastMX) * 0.005
    this.camPhi = Math.max(0.05, Math.min(Math.PI - 0.05, this.camPhi + (e.clientY - this.lastMY) * 0.005))
    this.lastMX = e.clientX; this.lastMY = e.clientY
  }

  private onPointerUp = () => { this.dragging = false }

  private onWheel = (e: WheelEvent) => {
    this.camDist = Math.max(5, Math.min(300, this.camDist + e.deltaY * 0.05))
  }

  setMeshCount(count: number) {
    if (this.useCase !== 'boxes') return

    while (this.cubes.length > count) {
      const cube = this.cubes.pop()!
      this.renderer.removeNode(cube.node)
    }

    const spread = 50
    while (this.cubes.length < count) {
      const i = this.cubes.length
      const node = this.renderer.createNode()
      node.mesh = this.boxMesh
      node.x = (Math.random() - 0.5) * spread
      node.y = (Math.random() - 0.5) * spread
      node.z = (Math.random() - 0.5) * spread
      node.rx = Math.random() * Math.PI * 2
      node.ry = Math.random() * Math.PI * 2
      const [r, g, b] = hslToRgb((i / Math.max(count, 1) + Math.random() * 0.05) % 1, 0.7, 0.55)
      node.setColor(r, g, b)
      node.updateTransform()

      this.cubes.push({
        node,
        rxSpeed: 0.5 + Math.random() * 2,
        rySpeed: 0.5 + Math.random() * 2,
      })
    }
  }

  setShadows(_enabled: boolean) {}

  render(dt: number) {
    if (this.useCase !== 'boxes' || this.cubes.length === 0) return

    // Update orbit camera position
    const sp = Math.sin(this.camPhi)
    this.camera.position[0] = this.camDist * sp * Math.sin(this.camTheta)
    this.camera.position[1] = this.camDist * Math.cos(this.camPhi)
    this.camera.position[2] = this.camDist * sp * Math.cos(this.camTheta)

    // Update cube rotations and transforms
    const cubes = this.cubes
    const n = cubes.length
    for (let i = 0; i < n; i++) {
      const c = cubes[i]
      const node = c.node
      node.rx += c.rxSpeed * dt
      node.ry += c.rySpeed * dt
      node.updateTransform()
    }

    this.renderer.render(this.camera)
  }

  dispose() {
    this.canvas.removeEventListener('pointerdown', this.onPointerDown)
    this.canvas.removeEventListener('pointermove', this.onPointerMove)
    this.canvas.removeEventListener('pointerup', this.onPointerUp)
    this.canvas.removeEventListener('wheel', this.onWheel)
    this.boxMesh?.destroy()
    this.renderer?.dispose()
    this.cubes = []
  }

  getInfo() {
    return 'Experiment C (WebGPU engine)'
  }
}
