import type { EngineAdapter, BackendType, UseCase } from '../types'

// --- WGSL Shader ---

const SHADER = /* wgsl */ `
struct Camera { viewProj: mat4x4f }
struct Instance { model: mat4x4f, color: vec4f }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> instance: Instance;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) normal: vec3f,
  @location(1) color: vec3f,
}

@vertex fn vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> VSOut {
  var out: VSOut;
  out.pos = camera.viewProj * instance.model * vec4f(position, 1.0);
  out.normal = (instance.model * vec4f(normal, 0.0)).xyz;
  out.color = instance.color.rgb;
  return out;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let lightDir = normalize(vec3f(30.0, 50.0, 30.0));
  let ambient = vec3f(0.25);
  let diffuse = max(dot(n, lightDir), 0.0) * 1.5;
  return vec4f(in.color * (ambient + vec3f(diffuse)), 1.0);
}
`

// --- Box geometry: 36 vertices, interleaved pos(3) + normal(3) ---

// prettier-ignore
function createBoxGeometry(): Float32Array {
  return new Float32Array([
    // +Z face
    -0.5,-0.5, 0.5,  0, 0, 1,   0.5,-0.5, 0.5,  0, 0, 1,   0.5, 0.5, 0.5,  0, 0, 1,
    -0.5,-0.5, 0.5,  0, 0, 1,   0.5, 0.5, 0.5,  0, 0, 1,  -0.5, 0.5, 0.5,  0, 0, 1,
    // -Z face
     0.5,-0.5,-0.5,  0, 0,-1,  -0.5,-0.5,-0.5,  0, 0,-1,  -0.5, 0.5,-0.5,  0, 0,-1,
     0.5,-0.5,-0.5,  0, 0,-1,  -0.5, 0.5,-0.5,  0, 0,-1,   0.5, 0.5,-0.5,  0, 0,-1,
    // +X face
     0.5,-0.5, 0.5,  1, 0, 0,   0.5,-0.5,-0.5,  1, 0, 0,   0.5, 0.5,-0.5,  1, 0, 0,
     0.5,-0.5, 0.5,  1, 0, 0,   0.5, 0.5,-0.5,  1, 0, 0,   0.5, 0.5, 0.5,  1, 0, 0,
    // -X face
    -0.5,-0.5,-0.5, -1, 0, 0,  -0.5,-0.5, 0.5, -1, 0, 0,  -0.5, 0.5, 0.5, -1, 0, 0,
    -0.5,-0.5,-0.5, -1, 0, 0,  -0.5, 0.5, 0.5, -1, 0, 0,  -0.5, 0.5,-0.5, -1, 0, 0,
    // +Y face
    -0.5, 0.5, 0.5,  0, 1, 0,   0.5, 0.5, 0.5,  0, 1, 0,   0.5, 0.5,-0.5,  0, 1, 0,
    -0.5, 0.5, 0.5,  0, 1, 0,   0.5, 0.5,-0.5,  0, 1, 0,  -0.5, 0.5,-0.5,  0, 1, 0,
    // -Y face
    -0.5,-0.5,-0.5,  0,-1, 0,   0.5,-0.5,-0.5,  0,-1, 0,   0.5,-0.5, 0.5,  0,-1, 0,
    -0.5,-0.5,-0.5,  0,-1, 0,   0.5,-0.5, 0.5,  0,-1, 0,  -0.5,-0.5, 0.5,  0,-1, 0,
  ])
}

// --- Constants ---

const INITIAL_CAPACITY = 10000
const UBO_STRIDE = 256 // must be >= minUniformBufferOffsetAlignment
const FLOAT_STRIDE = UBO_STRIDE / 4

// --- Minimal mat4 math (column-major Float32Array) ---

function mat4Perspective(out: Float32Array, fov: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fov / 2)
  const nf = 1 / (near - far)
  out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0
  out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0
  out[8] = 0; out[9] = 0; out[10] = (far + near) * nf; out[11] = -1
  out[12] = 0; out[13] = 0; out[14] = 2 * far * near * nf; out[15] = 0
}

function mat4LookAt(out: Float32Array, eye: number[], center: number[], up: number[]) {
  let fx = center[0] - eye[0], fy = center[1] - eye[1], fz = center[2] - eye[2]
  let len = 1 / Math.sqrt(fx * fx + fy * fy + fz * fz)
  fx *= len; fy *= len; fz *= len
  let rx = fy * up[2] - fz * up[1], ry = fz * up[0] - fx * up[2], rz = fx * up[1] - fy * up[0]
  len = 1 / Math.sqrt(rx * rx + ry * ry + rz * rz)
  rx *= len; ry *= len; rz *= len
  const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx
  out[0] = rx;  out[1] = ux;  out[2] = -fx; out[3] = 0
  out[4] = ry;  out[5] = uy;  out[6] = -fy; out[7] = 0
  out[8] = rz;  out[9] = uz;  out[10] = -fz; out[11] = 0
  out[12] = -(rx * eye[0] + ry * eye[1] + rz * eye[2])
  out[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2])
  out[14] = fx * eye[0] + fy * eye[1] + fz * eye[2]
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

// --- Mesh handle: views into the staging buffer ---

interface MeshHandle {
  modelMatrix: Float32Array // 16-float subarray
  color: Float32Array       // 4-float subarray
}

// --- Engine ---

export class ExperimentAAdapter implements EngineAdapter {
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private pipeline!: GPURenderPipeline
  private depthTexture!: GPUTexture
  private vertexBuffer!: GPUBuffer
  private vertexCount = 0
  private cameraBuffer!: GPUBuffer
  private instanceBuffer!: GPUBuffer
  private cameraBindGroup!: GPUBindGroup
  private instanceBindGroup!: GPUBindGroup
  private instanceLayout!: GPUBindGroupLayout

  private meshes: MeshHandle[] = []
  private meshCount = 0
  private capacity = INITIAL_CAPACITY
  private staging = new ArrayBuffer(INITIAL_CAPACITY * UBO_STRIDE)
  private stagingF32 = new Float32Array(this.staging)

  private canvas!: HTMLCanvasElement
  private vpMatrix = new Float32Array(16)
  private viewMatrix = new Float32Array(16)
  private projMatrix = new Float32Array(16)

  // Orbit camera (spherical coordinates)
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
  private animStates: Array<{ x: number; y: number; z: number; rx: number; ry: number }> = []

  async init(canvas: HTMLCanvasElement, backend: BackendType, useCase: UseCase) {
    if (backend !== 'webgpu') throw new Error('Experiment A supports WebGPU only')
    this.canvas = canvas
    this.useCase = useCase

    if (!navigator.gpu) throw new Error('WebGPU not supported')
    const gpuAdapter = await navigator.gpu.requestAdapter()
    if (!gpuAdapter) throw new Error('No WebGPU adapter found')
    this.device = await gpuAdapter.requestDevice() as GPUDevice

    this.context = canvas.getContext('webgpu')!
    const format = navigator.gpu.getPreferredCanvasFormat()

    const dpr = devicePixelRatio
    canvas.width = canvas.clientWidth * dpr
    canvas.height = canvas.clientHeight * dpr

    this.context.configure({ device: this.device, format, alphaMode: 'premultiplied' })

    // Camera uniform (mat4x4f = 64 bytes)
    this.cameraBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Per-instance dynamic uniform buffer
    this.instanceBuffer = this.device.createBuffer({
      size: this.capacity * UBO_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Layouts
    const cameraLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
    })
    this.instanceLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } }],
    })

    this.cameraBindGroup = this.device.createBindGroup({
      layout: cameraLayout,
      entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }],
    })
    this.instanceBindGroup = this.device.createBindGroup({
      layout: this.instanceLayout,
      entries: [{ binding: 0, resource: { buffer: this.instanceBuffer, size: 80 } }],
    })

    // Depth texture
    this.ensureDepthTexture()

    // Pipeline (vertex layout: stride=24, pos float32x3 + normal float32x3)
    const shaderModule = this.device.createShaderModule({ code: SHADER })
    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [cameraLayout, this.instanceLayout] }),
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
      fragment: {
        module: shaderModule,
        entryPoint: 'fs',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    })

    // Orbit controls
    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerup', this.onPointerUp)
    canvas.addEventListener('wheel', this.onWheel, { passive: true })
    window.addEventListener('resize', this.onResize)
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

  setGeometry(vertices: Float32Array, vertexCount: number) {
    if (this.vertexBuffer) this.vertexBuffer.destroy()
    this.vertexBuffer = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices)
    this.vertexCount = vertexCount
  }

  // --- General-purpose: mesh capacity ---

  private ensureCapacity(count: number) {
    if (count <= this.capacity) return
    this.capacity = count
    const newStaging = new ArrayBuffer(count * UBO_STRIDE)
    const newF32 = new Float32Array(newStaging)
    // Copy existing data (preserves transforms and colors)
    newF32.set(new Float32Array(this.staging, 0, Math.min(this.staging.byteLength, newStaging.byteLength) / 4))
    this.staging = newStaging
    this.stagingF32 = newF32
    // Rebuild mesh handle views into new buffer
    for (let i = 0; i < this.meshes.length; i++) {
      const o = i * FLOAT_STRIDE
      this.meshes[i].modelMatrix = newF32.subarray(o, o + 16)
      this.meshes[i].color = newF32.subarray(o + 16, o + 20)
    }
    // Recreate GPU buffer + bind group
    this.instanceBuffer.destroy()
    this.instanceBuffer = this.device.createBuffer({
      size: count * UBO_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.instanceBindGroup = this.device.createBindGroup({
      layout: this.instanceLayout,
      entries: [{ binding: 0, resource: { buffer: this.instanceBuffer, size: 80 } }],
    })
  }

  // --- EngineAdapter interface ---

  setMeshCount(count: number) {
    if (this.useCase !== 'boxes') return

    // Set box geometry on first call
    if (!this.vertexCount) {
      this.setGeometry(createBoxGeometry(), 36)
    }

    this.ensureCapacity(count)

    const spread = 50
    while (this.meshes.length < count) {
      const i = this.meshes.length
      const o = i * FLOAT_STRIDE
      const handle: MeshHandle = {
        modelMatrix: this.stagingF32.subarray(o, o + 16),
        color: this.stagingF32.subarray(o + 16, o + 20),
      }
      handle.color[0] = Math.random()
      handle.color[1] = Math.random()
      handle.color[2] = Math.random()
      handle.color[3] = 1
      this.meshes.push(handle)

      // Benchmark-specific: box animation state
      this.animStates.push({
        x: (Math.random() - 0.5) * spread,
        y: (Math.random() - 0.5) * spread,
        z: (Math.random() - 0.5) * spread,
        rx: 0, ry: 0,
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

    this.onResize() // cheap no-op if size unchanged thanks to ensureDepthTexture guard

    const w = this.canvas.width
    const h = this.canvas.height

    // Camera VP matrix
    const eye = [
      this.camDist * Math.sin(this.camPhi) * Math.sin(this.camTheta),
      this.camDist * Math.cos(this.camPhi),
      this.camDist * Math.sin(this.camPhi) * Math.cos(this.camTheta),
    ]
    mat4Perspective(this.projMatrix, Math.PI / 3, w / h, 0.1, 1000)
    mat4LookAt(this.viewMatrix, eye, [0, 0, 0], [0, 1, 0])
    mat4Mul(this.vpMatrix, this.projMatrix, this.viewMatrix)
    this.device.queue.writeBuffer(this.cameraBuffer, 0, this.vpMatrix)

    // Benchmark-specific: animate boxes (writes directly into staging via mesh handles)
    const speed = 1.0
    for (let i = 0; i < this.meshCount; i++) {
      const s = this.animStates[i]
      s.rx += speed * dt
      s.ry += speed * dt * 0.7
      mat4RotYXTranslation(this.meshes[i].modelMatrix, s.rx, s.ry, s.x, s.y, s.z)
    }

    // General: upload all mesh data (single writeBuffer for all instances)
    this.device.queue.writeBuffer(this.instanceBuffer, 0, this.staging, 0, this.meshCount * UBO_STRIDE)

    // General: encode render pass
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
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
    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.setBindGroup(0, this.cameraBindGroup)

    for (let i = 0; i < this.meshCount; i++) {
      pass.setBindGroup(1, this.instanceBindGroup, [i * UBO_STRIDE])
      pass.draw(this.vertexCount)
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
    this.vertexBuffer?.destroy()
    this.cameraBuffer?.destroy()
    this.instanceBuffer?.destroy()
    this.depthTexture?.destroy()
    this.device?.destroy()
    this.meshes = []
    this.animStates = []
  }

  getInfo() {
    return 'Experiment A (WebGPU raw)'
  }
}
