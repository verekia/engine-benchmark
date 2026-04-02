import type { EngineAdapter, BackendType, UseCase } from '../types'

// --- WGSL Shader ---

const SHADER = /* wgsl */ `
struct Camera { viewProj: mat4x4f }
struct Cube { model: mat4x4f, color: vec4f }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> cube: Cube;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) normal: vec3f,
  @location(1) color: vec3f,
}

@vertex fn vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> VSOut {
  var out: VSOut;
  out.pos = camera.viewProj * cube.model * vec4f(position, 1.0);
  out.normal = (cube.model * vec4f(normal, 0.0)).xyz;
  out.color = cube.color.rgb;
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

// --- Cube geometry: 36 vertices, 6 floats each (pos xyz + normal xyz) ---

// prettier-ignore
const CUBE_VERTS = new Float32Array([
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

// --- Constants ---

const MAX_CUBES = 10000
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

// --- Cube state ---

interface CubeState {
  x: number; y: number; z: number
  rx: number; ry: number
  r: number; g: number; b: number
}

// --- Engine ---

export class ExperimentAAdapter implements EngineAdapter {
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private pipeline!: GPURenderPipeline
  private depthTexture!: GPUTexture
  private vertexBuffer!: GPUBuffer
  private cameraBuffer!: GPUBuffer
  private cubeBuffer!: GPUBuffer
  private cameraBindGroup!: GPUBindGroup
  private cubeBindGroup!: GPUBindGroup

  private cubes: CubeState[] = []
  private staging = new ArrayBuffer(MAX_CUBES * UBO_STRIDE)
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

    // Vertex buffer
    this.vertexBuffer = this.device.createBuffer({
      size: CUBE_VERTS.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.vertexBuffer, 0, CUBE_VERTS)

    // Camera uniform (mat4x4f = 64 bytes)
    this.cameraBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Per-cube dynamic uniform buffer
    this.cubeBuffer = this.device.createBuffer({
      size: MAX_CUBES * UBO_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Layouts
    const cameraLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
    })
    const cubeLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } }],
    })

    this.cameraBindGroup = this.device.createBindGroup({
      layout: cameraLayout,
      entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }],
    })
    this.cubeBindGroup = this.device.createBindGroup({
      layout: cubeLayout,
      entries: [{ binding: 0, resource: { buffer: this.cubeBuffer, size: 80 } }],
    })

    // Depth texture
    this.ensureDepthTexture()

    // Pipeline
    const shaderModule = this.device.createShaderModule({ code: SHADER })
    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [cameraLayout, cubeLayout] }),
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

  // --- EngineAdapter interface ---

  setMeshCount(count: number) {
    if (this.useCase !== 'boxes') return
    if (count > MAX_CUBES) count = MAX_CUBES

    if (this.cubes.length > count) {
      this.cubes.length = count
    }

    const spread = 50
    while (this.cubes.length < count) {
      this.cubes.push({
        x: (Math.random() - 0.5) * spread,
        y: (Math.random() - 0.5) * spread,
        z: (Math.random() - 0.5) * spread,
        rx: 0, ry: 0,
        r: Math.random(), g: Math.random(), b: Math.random(),
      })
    }
  }

  setShadows(_enabled: boolean) {
    // Shadows not implemented in this minimal engine
  }

  render(dt: number) {
    if (this.useCase !== 'boxes' || this.cubes.length === 0) return

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

    // Update per-cube transforms into staging buffer
    const f = this.stagingF32
    const n = this.cubes.length
    const speed = 1.0

    for (let i = 0; i < n; i++) {
      const c = this.cubes[i]
      c.rx += speed * dt
      c.ry += speed * dt * 0.7

      const sx = Math.sin(c.rx), cx = Math.cos(c.rx)
      const sy = Math.sin(c.ry), cy = Math.cos(c.ry)
      const o = i * FLOAT_STRIDE

      // Model = Translation * RotY * RotX  (column-major)
      f[o     ] = cy;      f[o +  1] = 0;   f[o +  2] = -sy;     f[o +  3] = 0
      f[o +  4] = sy * sx; f[o +  5] = cx;  f[o +  6] = cy * sx; f[o +  7] = 0
      f[o +  8] = sy * cx; f[o +  9] = -sx; f[o + 10] = cy * cx; f[o + 11] = 0
      f[o + 12] = c.x;     f[o + 13] = c.y; f[o + 14] = c.z;     f[o + 15] = 1

      // Color (vec4f at byte offset 64)
      f[o + 16] = c.r; f[o + 17] = c.g; f[o + 18] = c.b; f[o + 19] = 1
    }

    this.device.queue.writeBuffer(this.cubeBuffer, 0, this.staging, 0, n * UBO_STRIDE)

    // Encode render pass
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

    for (let i = 0; i < n; i++) {
      pass.setBindGroup(1, this.cubeBindGroup, [i * UBO_STRIDE])
      pass.draw(36)
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
    this.cubeBuffer?.destroy()
    this.depthTexture?.destroy()
    this.device?.destroy()
    this.cubes = []
  }

  getInfo() {
    return 'Experiment A (WebGPU raw)'
  }
}
