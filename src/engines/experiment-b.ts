import type { EngineAdapter, BackendType, UseCase } from '../types'

// --- WGSL Shader (storage buffer for per-instance data, Lambert lighting) ---

const SHADER = /* wgsl */ `
struct Scene {
  viewProj: mat4x4f,
  lightDir: vec4f,
  ambient: vec4f,
  lightCol: vec4f,
}

struct Instance {
  col0: vec4f,
  col1: vec4f,
  col2: vec4f,
  col3: vec4f,
  color: vec4f,
}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;

struct V {
  @builtin(position) pos: vec4f,
  @location(0) norm: vec3f,
  @location(1) col: vec3f,
}

@vertex fn vs(
  @location(0) p: vec3f,
  @location(1) n: vec3f,
  @builtin(instance_index) i: u32,
) -> V {
  let inst = instances[i];
  let m = mat4x4f(inst.col0, inst.col1, inst.col2, inst.col3);
  let wp = m * vec4f(p, 1.0);
  var o: V;
  o.pos = scene.viewProj * wp;
  o.norm = normalize((m * vec4f(n, 0.0)).xyz);
  o.col = inst.color.rgb;
  return o;
}

@fragment fn fs(v: V) -> @location(0) vec4f {
  let n = normalize(v.norm);
  let d = max(dot(n, scene.lightDir.xyz), 0.0);
  let c = v.col * (scene.ambient.rgb + scene.lightCol.rgb * d);
  return vec4f(c, 1.0);
}
`

// --- Box geometry: 24 verts (indexed), interleaved pos(3) + normal(3) ---

function createBoxGeometry() {
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

// --- Minimal mat4 math (column-major) ---

function mat4Perspective(out: Float32Array, fov: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fov / 2)
  const nf = 1 / (near - far)
  out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0
  out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0
  out[8] = 0; out[9] = 0; out[10] = far * nf; out[11] = -1
  out[12] = 0; out[13] = 0; out[14] = near * far * nf; out[15] = 0
}

function mat4LookAt(out: Float32Array, ex: number, ey: number, ez: number, cx: number, cy: number, cz: number) {
  let fx = cx - ex, fy = cy - ey, fz = cz - ez
  let il = 1 / Math.sqrt(fx * fx + fy * fy + fz * fz)
  fx *= il; fy *= il; fz *= il
  let rx = fz, rz = -fx
  il = 1 / Math.sqrt(rx * rx + rz * rz)
  rx *= il; rz *= il
  const ux = -fy * rz, uy = rz * fx - rx * fz, uz = rx * fy
  out[0] = rx;  out[1] = ux;  out[2] = -fx; out[3] = 0
  out[4] = 0;   out[5] = uy;  out[6] = -fy; out[7] = 0
  out[8] = rz;  out[9] = uz;  out[10] = -fz; out[11] = 0
  out[12] = -(rx * ex + rz * ez)
  out[13] = -(ux * ex + uy * ey + uz * ez)
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

function mat4RotYXTranslation(out: Float32Array, rx: number, ry: number, tx: number, ty: number, tz: number) {
  const sx = Math.sin(rx), cx = Math.cos(rx)
  const sy = Math.sin(ry), cy = Math.cos(ry)
  out[0] = cy;      out[1] = 0;   out[2] = -sy;     out[3] = 0
  out[4] = sy * sx; out[5] = cx;  out[6] = cy * sx;  out[7] = 0
  out[8] = sy * cx; out[9] = -sx; out[10] = cy * cx; out[11] = 0
  out[12] = tx;     out[13] = ty; out[14] = tz;      out[15] = 1
}

// --- Constants ---

const FLOATS_PER_MESH = 20 // mat4 (16) + color (4)
const INITIAL_CAPACITY = 1000

// --- Mesh handle: views into the GPU data buffer ---

interface MeshHandle {
  modelMatrix: Float32Array // 16-float subarray
  color: Float32Array       // 4-float subarray
}

// --- Utility ---

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((h * 6) % 2) - 1))
  const m = l - c * 0.5
  let r = 0, g = 0, b = 0
  const sector = (h * 6) | 0
  if (sector === 0) { r = c; g = x }
  else if (sector === 1) { r = x; g = c }
  else if (sector === 2) { g = c; b = x }
  else if (sector === 3) { g = x; b = c }
  else if (sector === 4) { r = x; b = c }
  else { r = c; b = x }
  return [r + m, g + m, b + m]
}

// --- Engine ---

export class ExperimentBAdapter implements EngineAdapter {
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private pipeline!: GPURenderPipeline
  private depthTexture!: GPUTexture
  private vertBuf!: GPUBuffer
  private idxBuf!: GPUBuffer
  private indexCount = 0
  private sceneBuf!: GPUBuffer
  private instanceBuf!: GPUBuffer
  private bindGroupLayout!: GPUBindGroupLayout
  private bindGroup!: GPUBindGroup

  private meshes: MeshHandle[] = []
  private meshCount = 0
  private gpuData = new Float32Array(0)
  private instanceBufSize = 0
  private sceneData = new Float32Array(28) // viewProj(16) + lightDir(4) + ambient(4) + lightCol(4)

  private canvas!: HTMLCanvasElement
  private vpMatrix = new Float32Array(16)
  private viewMatrix = new Float32Array(16)
  private projMatrix = new Float32Array(16)

  // Orbit camera
  private camTheta = 0
  private camPhi = Math.acos(40 / 72)
  private camDist = 72
  private dragging = false
  private lastMX = 0
  private lastMY = 0

  private useCase: UseCase = 'boxes'
  private depthW = 0
  private depthH = 0

  // Benchmark-specific animation state
  private animStates: Array<{
    x: number; y: number; z: number
    rx: number; ry: number
    rxSpeed: number; rySpeed: number
  }> = []

  async init(canvas: HTMLCanvasElement, backend: BackendType, useCase: UseCase) {
    if (backend !== 'webgpu') throw new Error('Experiment B supports WebGPU only')
    this.canvas = canvas
    this.useCase = useCase

    if (!navigator.gpu) throw new Error('WebGPU not supported')
    const gpuAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!gpuAdapter) throw new Error('No WebGPU adapter found')
    this.device = await gpuAdapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: gpuAdapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: gpuAdapter.limits.maxBufferSize,
      },
    }) as GPUDevice

    this.context = canvas.getContext('webgpu')!
    const format = navigator.gpu.getPreferredCanvasFormat()

    const dpr = devicePixelRatio
    canvas.width = canvas.clientWidth * dpr
    canvas.height = canvas.clientHeight * dpr

    this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' })

    // Scene uniform (128 bytes, rounded up from 112)
    this.sceneBuf = this.device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })

    // Lighting constants
    const ldx = 0.5, ldy = 1.0, ldz = 0.3
    const ldLen = 1 / Math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz)
    this.sceneData[16] = ldx * ldLen; this.sceneData[17] = ldy * ldLen; this.sceneData[18] = ldz * ldLen; this.sceneData[19] = 0
    this.sceneData[20] = 0.15; this.sceneData[21] = 0.15; this.sceneData[22] = 0.18; this.sceneData[23] = 0 // ambient
    this.sceneData[24] = 1.0; this.sceneData[25] = 0.95; this.sceneData[26] = 0.9; this.sceneData[27] = 0 // light color

    // Initial instance storage buffer
    this.instanceBufSize = INITIAL_CAPACITY * FLOATS_PER_MESH * 4
    this.instanceBuf = this.device.createBuffer({
      size: this.instanceBufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    // Pipeline (vertex layout: stride=24, pos float32x3 + normal float32x3)
    const shaderModule = this.device.createShaderModule({ code: SHADER })
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    })

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: {
        module: shaderModule,
        entryPoint: 'vs',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: shaderModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    })

    this.rebuildBindGroup()
    this.ensureDepthTexture()

    // Orbit controls
    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerup', this.onPointerUp)
    canvas.addEventListener('wheel', this.onWheel, { passive: true })
    window.addEventListener('resize', this.onResize)
  }

  private rebuildBindGroup() {
    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.sceneBuf } },
        { binding: 1, resource: { buffer: this.instanceBuf } },
      ],
    })
  }

  private ensureDepthTexture() {
    const w = this.canvas.width
    const h = this.canvas.height
    if (w === this.depthW && h === this.depthH) return
    this.depthW = w
    this.depthH = h
    if (this.depthTexture) this.depthTexture.destroy()
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
  }

  // --- Orbit controls ---

  private onPointerDown = (e: PointerEvent) => {
    this.dragging = true
    this.lastMX = e.clientX
    this.lastMY = e.clientY
    this.canvas.setPointerCapture(e.pointerId)
  }

  private onPointerMove = (e: PointerEvent) => {
    if (!this.dragging) return
    const dx = e.clientX - this.lastMX
    const dy = e.clientY - this.lastMY
    this.lastMX = e.clientX
    this.lastMY = e.clientY
    this.camTheta -= dx * 0.005
    this.camPhi = Math.max(0.05, Math.min(Math.PI - 0.05, this.camPhi + dy * 0.005))
  }

  private onPointerUp = () => { this.dragging = false }

  private onWheel = (e: WheelEvent) => {
    this.camDist = Math.max(5, Math.min(300, this.camDist + e.deltaY * 0.05))
  }

  private onResize = () => {
    const dpr = devicePixelRatio
    this.canvas.width = this.canvas.clientWidth * dpr
    this.canvas.height = this.canvas.clientHeight * dpr
    this.ensureDepthTexture()
  }

  // --- General-purpose: geometry ---

  setGeometry(vertices: Float32Array, indices: Uint16Array) {
    if (this.vertBuf) this.vertBuf.destroy()
    if (this.idxBuf) this.idxBuf.destroy()
    this.vertBuf = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.vertBuf, 0, vertices)
    this.idxBuf = this.device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.idxBuf, 0, indices)
    this.indexCount = indices.length
  }

  // --- General-purpose: mesh capacity ---

  private ensureCapacity(count: number) {
    // Resize CPU data array
    if (count * FLOATS_PER_MESH > this.gpuData.length) {
      const newData = new Float32Array(count * FLOATS_PER_MESH)
      newData.set(this.gpuData)
      this.gpuData = newData
      // Rebuild mesh handle views into new buffer
      for (let i = 0; i < this.meshes.length; i++) {
        const o = i * FLOATS_PER_MESH
        this.meshes[i].modelMatrix = newData.subarray(o, o + 16)
        this.meshes[i].color = newData.subarray(o + 16, o + 20)
      }
    }

    // Resize GPU storage buffer
    const needed = count * FLOATS_PER_MESH * 4
    if (needed > this.instanceBufSize) {
      this.instanceBuf.destroy()
      this.instanceBufSize = needed
      this.instanceBuf = this.device.createBuffer({
        size: needed,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      this.rebuildBindGroup()
    }
  }

  // --- EngineAdapter interface ---

  setMeshCount(count: number) {
    if (this.useCase !== 'boxes') return

    // Set box geometry on first call
    if (!this.indexCount) {
      const { vertices, indices } = createBoxGeometry()
      this.setGeometry(vertices, indices)
    }

    this.ensureCapacity(count)

    const spread = 50
    while (this.meshes.length < count) {
      const i = this.meshes.length
      const total = Math.max(count, 1)
      const o = i * FLOATS_PER_MESH
      const handle: MeshHandle = {
        modelMatrix: this.gpuData.subarray(o, o + 16),
        color: this.gpuData.subarray(o + 16, o + 20),
      }
      const [r, g, b] = hslToRgb((i / total + Math.random() * 0.05) % 1, 0.7, 0.55)
      handle.color[0] = r
      handle.color[1] = g
      handle.color[2] = b
      handle.color[3] = 1
      this.meshes.push(handle)

      // Benchmark-specific: box animation state
      this.animStates.push({
        x: (Math.random() - 0.5) * spread,
        y: (Math.random() - 0.5) * spread,
        z: (Math.random() - 0.5) * spread,
        rx: Math.random() * Math.PI * 2,
        ry: Math.random() * Math.PI * 2,
        rxSpeed: 0.5 + Math.random() * 2,
        rySpeed: 0.5 + Math.random() * 2,
      })
    }

    if (this.meshes.length > count) {
      this.meshes.length = count
      this.animStates.length = count
    }

    this.meshCount = count
  }

  setShadows(_enabled: boolean) {
    // Shadows not implemented in this minimal engine
  }

  render(dt: number) {
    if (this.useCase !== 'boxes' || this.meshCount === 0) return

    this.onResize()

    const w = this.canvas.width
    const h = this.canvas.height
    const n = this.meshCount

    // Camera
    const eye = [
      this.camDist * Math.sin(this.camPhi) * Math.sin(this.camTheta),
      this.camDist * Math.cos(this.camPhi),
      this.camDist * Math.sin(this.camPhi) * Math.cos(this.camTheta),
    ]
    mat4Perspective(this.projMatrix, Math.PI / 3, w / h, 0.1, 1000)
    mat4LookAt(this.viewMatrix, eye[0], eye[1], eye[2], 0, 0, 0)
    mat4Mul(this.vpMatrix, this.projMatrix, this.viewMatrix)

    // Write viewProj into scene data
    this.sceneData.set(this.vpMatrix, 0)
    this.device.queue.writeBuffer(this.sceneBuf, 0, this.sceneData)

    // Benchmark-specific: animate boxes (writes directly into gpuData via mesh handles)
    for (let i = 0; i < n; i++) {
      const s = this.animStates[i]
      s.rx += s.rxSpeed * dt
      s.ry += s.rySpeed * dt
      mat4RotYXTranslation(this.meshes[i].modelMatrix, s.rx, s.ry, s.x, s.y, s.z)
    }

    // General: upload all mesh data (single writeBuffer for all instances)
    this.device.queue.writeBuffer(this.instanceBuf, 0, this.gpuData.buffer, 0, n * FLOATS_PER_MESH * 4)

    // General: encode render pass
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.05, g: 0.05, b: 0.07, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })

    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.setVertexBuffer(0, this.vertBuf)
    pass.setIndexBuffer(this.idxBuf, 'uint16')

    for (let i = 0; i < n; i++) {
      pass.drawIndexed(this.indexCount, 1, 0, 0, i)
    }

    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  dispose() {
    this.canvas.removeEventListener('pointerdown', this.onPointerDown)
    this.canvas.removeEventListener('pointermove', this.onPointerMove)
    this.canvas.removeEventListener('pointerup', this.onPointerUp)
    this.canvas.removeEventListener('wheel', this.onWheel)
    window.removeEventListener('resize', this.onResize)
    this.vertBuf?.destroy()
    this.idxBuf?.destroy()
    this.sceneBuf?.destroy()
    this.instanceBuf?.destroy()
    this.depthTexture?.destroy()
    this.device?.destroy()
    this.meshes = []
    this.animStates = []
  }

  getInfo() {
    return 'Experiment B (WebGPU storage)'
  }
}
