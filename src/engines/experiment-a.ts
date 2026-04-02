import type { EngineAdapter, BackendType, UseCase } from '../types'
import { createUniqueTetrahedronSpec, type BufferGeometryData } from '../benchmark-scene'

const SHADER = /* wgsl */ `
struct Scene {
  viewProj: mat4x4f,
  lightDir: vec4f,
  ambient: vec4f,
  lightColor: vec4f,
}

struct ObjectData {
  model: mat4x4f,
  color: vec4f,
}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var<uniform> objectData: ObjectData;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) normal: vec3f,
  @location(1) color: vec3f,
}

@vertex fn vs(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
) -> VSOut {
  var out: VSOut;
  out.pos = scene.viewProj * objectData.model * vec4f(position, 1.0);
  out.normal = normalize((objectData.model * vec4f(normal, 0.0)).xyz);
  out.color = objectData.color.rgb;
  return out;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let light = max(dot(normalize(in.normal), scene.lightDir.xyz), 0.0);
  let color = in.color * (scene.ambient.rgb + scene.lightColor.rgb * light);
  return vec4f(color, 1.0);
}
`

function createBoxGeometryData(): BufferGeometryData {
  const positions = new Float32Array([
    // +X
     0.5, -0.5, -0.5,   0.5,  0.5, -0.5,   0.5,  0.5,  0.5,   0.5, -0.5,  0.5,
    // -X
    -0.5, -0.5,  0.5,  -0.5,  0.5,  0.5,  -0.5,  0.5, -0.5,  -0.5, -0.5, -0.5,
    // +Y
    -0.5,  0.5,  0.5,   0.5,  0.5,  0.5,   0.5,  0.5, -0.5,  -0.5,  0.5, -0.5,
    // -Y
    -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5, -0.5,  0.5,  -0.5, -0.5,  0.5,
    // +Z
    -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,
    // -Z
     0.5, -0.5, -0.5,  -0.5, -0.5, -0.5,  -0.5,  0.5, -0.5,   0.5,  0.5, -0.5,
  ])

  const normals = new Float32Array([
    // +X
     1, 0, 0,   1, 0, 0,   1, 0, 0,   1, 0, 0,
    // -X
    -1, 0, 0,  -1, 0, 0,  -1, 0, 0,  -1, 0, 0,
    // +Y
     0, 1, 0,   0, 1, 0,   0, 1, 0,   0, 1, 0,
    // -Y
     0,-1, 0,   0,-1, 0,   0,-1, 0,   0,-1, 0,
    // +Z
     0, 0, 1,   0, 0, 1,   0, 0, 1,   0, 0, 1,
    // -Z
     0, 0,-1,   0, 0,-1,   0, 0,-1,   0, 0,-1,
  ])

  const indices = new Uint16Array([
    0, 1, 2, 0, 2, 3,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23,
  ])

  return { positions, normals, indices }
}

function mat4Perspective(out: Float32Array, fov: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fov / 2)
  const nf = 1 / (near - far)
  out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0
  out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0
  out[8] = 0; out[9] = 0; out[10] = far * nf; out[11] = -1
  out[12] = 0; out[13] = 0; out[14] = near * far * nf; out[15] = 0
}

function mat4LookAt(out: Float32Array, eye: number[], center: number[], up: number[]) {
  let fx = center[0] - eye[0], fy = center[1] - eye[1], fz = center[2] - eye[2]
  let len = 1 / Math.sqrt(fx * fx + fy * fy + fz * fz)
  fx *= len; fy *= len; fz *= len
  let rx = fy * up[2] - fz * up[1], ry = fz * up[0] - fx * up[2], rz = fx * up[1] - fy * up[0]
  len = 1 / Math.sqrt(rx * rx + ry * ry + rz * rz)
  rx *= len; ry *= len; rz *= len
  const sx = ry * fz - rz * fy, sy = rz * fx - rx * fz, sz = rx * fy - ry * fx
  out[0] = rx; out[1] = sx; out[2] = -fx; out[3] = 0
  out[4] = ry; out[5] = sy; out[6] = -fy; out[7] = 0
  out[8] = rz; out[9] = sz; out[10] = -fz; out[11] = 0
  out[12] = -(rx * eye[0] + ry * eye[1] + rz * eye[2])
  out[13] = -(sx * eye[0] + sy * eye[1] + sz * eye[2])
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

class BufferGeometry {
  readonly vertexBuffer: GPUBuffer
  readonly indexBuffer: GPUBuffer
  readonly indexCount: number

  constructor(device: GPUDevice, data: BufferGeometryData) {
    const vertexCount = data.positions.length / 3
    const interleaved = new Float32Array(vertexCount * 6)
    for (let i = 0; i < vertexCount; i++) {
      interleaved[i * 6] = data.positions[i * 3]
      interleaved[i * 6 + 1] = data.positions[i * 3 + 1]
      interleaved[i * 6 + 2] = data.positions[i * 3 + 2]
      interleaved[i * 6 + 3] = data.normals[i * 3]
      interleaved[i * 6 + 4] = data.normals[i * 3 + 1]
      interleaved[i * 6 + 5] = data.normals[i * 3 + 2]
    }

    this.indexCount = data.indices.length
    this.vertexBuffer = device.createBuffer({
      size: interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.vertexBuffer, 0, interleaved)

    this.indexBuffer = device.createBuffer({
      size: data.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.indexBuffer, 0, data.indices)
  }

  destroy() {
    this.vertexBuffer.destroy()
    this.indexBuffer.destroy()
  }
}

class LambertMaterial {
  readonly color: [number, number, number]

  constructor(color: [number, number, number]) {
    this.color = color
  }
}

class Mesh {
  x = 0; y = 0; z = 0
  rx = 0; ry = 0; rz = 0
  sx = 1; sy = 1; sz = 1
  rxSpeed = 0; rySpeed = 0; rzSpeed = 0

  constructor(
    readonly geometry: BufferGeometry,
    readonly material: LambertMaterial,
  ) {}

  writeObjectData(target: Float32Array, offset: number) {
    const cosX = Math.cos(this.rx), sinX = Math.sin(this.rx)
    const cosY = Math.cos(this.ry), sinY = Math.sin(this.ry)
    const cosZ = Math.cos(this.rz), sinZ = Math.sin(this.rz)

    target[offset] = (cosZ * cosY) * this.sx
    target[offset + 1] = (sinZ * cosY) * this.sx
    target[offset + 2] = (-sinY) * this.sx
    target[offset + 3] = 0
    target[offset + 4] = (cosZ * sinY * sinX - sinZ * cosX) * this.sy
    target[offset + 5] = (sinZ * sinY * sinX + cosZ * cosX) * this.sy
    target[offset + 6] = (cosY * sinX) * this.sy
    target[offset + 7] = 0
    target[offset + 8] = (cosZ * sinY * cosX + sinZ * sinX) * this.sz
    target[offset + 9] = (sinZ * sinY * cosX - cosZ * sinX) * this.sz
    target[offset + 10] = (cosY * cosX) * this.sz
    target[offset + 11] = 0
    target[offset + 12] = this.x
    target[offset + 13] = this.y
    target[offset + 14] = this.z
    target[offset + 15] = 1
    target[offset + 16] = this.material.color[0]
    target[offset + 17] = this.material.color[1]
    target[offset + 18] = this.material.color[2]
    target[offset + 19] = 1
  }
}

const INITIAL_CAPACITY = 1024
const OBJECT_FLOATS = 20
const SCENE_FLOATS = 28

export class ExperimentAAdapter implements EngineAdapter {
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private pipeline!: GPURenderPipeline
  private depthTexture!: GPUTexture
  private depthW = 0
  private depthH = 0
  private sceneBuffer!: GPUBuffer
  private objectBuffer!: GPUBuffer
  private sceneBindGroup!: GPUBindGroup
  private objectBindGroup!: GPUBindGroup
  private objectLayout!: GPUBindGroupLayout

  private canvas!: HTMLCanvasElement
  private useCase: UseCase = 'boxes'
  private boxGeometry: BufferGeometry | null = null
  private meshes: Mesh[] = []
  private capacity = INITIAL_CAPACITY
  private objectStride = 256
  private objectFloatStride = 64
  private objectStaging = new Float32Array(INITIAL_CAPACITY * 64)
  private sceneData = new Float32Array(SCENE_FLOATS)

  private viewMatrix = new Float32Array(16)
  private projMatrix = new Float32Array(16)
  private vpMatrix = new Float32Array(16)

  private camTheta = 0
  private camPhi = Math.acos(40 / 72)
  private camDist = 72
  private dragging = false
  private lastMX = 0
  private lastMY = 0

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

    const uniformAlign = this.device.limits.minUniformBufferOffsetAlignment
    this.objectStride = Math.ceil((OBJECT_FLOATS * 4) / uniformAlign) * uniformAlign
    this.objectFloatStride = this.objectStride / 4
    this.objectStaging = new Float32Array(this.capacity * this.objectFloatStride)

    this.sceneBuffer = this.device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.objectBuffer = this.device.createBuffer({
      size: this.capacity * this.objectStride,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.sceneData[16] = 0.5
    this.sceneData[17] = 1.0
    this.sceneData[18] = 0.3
    this.sceneData[20] = 0.18
    this.sceneData[21] = 0.18
    this.sceneData[22] = 0.2
    this.sceneData[24] = 1.0
    this.sceneData[25] = 0.95
    this.sceneData[26] = 0.9

    const sceneLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    })
    this.objectLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } }],
    })

    this.sceneBindGroup = this.device.createBindGroup({
      layout: sceneLayout,
      entries: [{ binding: 0, resource: { buffer: this.sceneBuffer } }],
    })
    this.objectBindGroup = this.device.createBindGroup({
      layout: this.objectLayout,
      entries: [{ binding: 0, resource: { buffer: this.objectBuffer, size: OBJECT_FLOATS * 4 } }],
    })

    const shader = this.device.createShaderModule({ code: SHADER })
    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, this.objectLayout] }),
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
      fragment: {
        module: shader,
        entryPoint: 'fs',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    })

    this.ensureDepthTexture()

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

  private ensureCapacity(count: number) {
    if (count <= this.capacity) return
    this.capacity = count
    this.objectStaging = new Float32Array(count * this.objectFloatStride)
    this.objectBuffer.destroy()
    this.objectBuffer = this.device.createBuffer({
      size: count * this.objectStride,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.objectBindGroup = this.device.createBindGroup({
      layout: this.objectLayout,
      entries: [{ binding: 0, resource: { buffer: this.objectBuffer, size: OBJECT_FLOATS * 4 } }],
    })
  }

  private destroyMesh(mesh: Mesh) {
    if (mesh.geometry !== this.boxGeometry) {
      mesh.geometry.destroy()
    }
  }

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

  private onPointerUp = () => {
    this.dragging = false
  }

  private onWheel = (e: WheelEvent) => {
    this.camDist = Math.max(5, Math.min(300, this.camDist + e.deltaY * 0.05))
  }

  private onResize = () => {
    const dpr = devicePixelRatio
    this.canvas.width = this.canvas.clientWidth * dpr
    this.canvas.height = this.canvas.clientHeight * dpr
    this.ensureDepthTexture()
  }

  setMeshCount(count: number) {
    if (this.useCase !== 'boxes' && this.useCase !== 'unique-tetrahedra') return
    this.ensureCapacity(count)

    if (this.useCase === 'boxes') {
      if (!this.boxGeometry) {
        this.boxGeometry = new BufferGeometry(this.device, createBoxGeometryData())
      }

      const spread = 50
      while (this.meshes.length < count) {
        const mesh = new Mesh(
          this.boxGeometry,
          new LambertMaterial([Math.random(), Math.random(), Math.random()]),
        )
        mesh.x = (Math.random() - 0.5) * spread
        mesh.y = (Math.random() - 0.5) * spread
        mesh.z = (Math.random() - 0.5) * spread
        mesh.rxSpeed = 1.0
        mesh.rySpeed = 0.7
        this.meshes.push(mesh)
      }
    } else {
      while (this.meshes.length < count) {
        const spec = createUniqueTetrahedronSpec(this.meshes.length)
        const mesh = new Mesh(
          new BufferGeometry(this.device, spec.geometry),
          new LambertMaterial(spec.material.color),
        )
        mesh.x = spec.position[0]
        mesh.y = spec.position[1]
        mesh.z = spec.position[2]
        mesh.rx = spec.rotation[0]
        mesh.ry = spec.rotation[1]
        mesh.rz = spec.rotation[2]
        mesh.rxSpeed = spec.rotationSpeed[0]
        mesh.rySpeed = spec.rotationSpeed[1]
        mesh.rzSpeed = spec.rotationSpeed[2]
        this.meshes.push(mesh)
      }
    }

    while (this.meshes.length > count) {
      this.destroyMesh(this.meshes.pop()!)
    }
  }

  setShadows(_enabled: boolean) {
    // Shadows not implemented in this minimal experiment.
  }

  render(dt: number) {
    if ((this.useCase !== 'boxes' && this.useCase !== 'unique-tetrahedra') || this.meshes.length === 0) return

    this.onResize()

    const eye = [
      this.camDist * Math.sin(this.camPhi) * Math.sin(this.camTheta),
      this.camDist * Math.cos(this.camPhi),
      this.camDist * Math.sin(this.camPhi) * Math.cos(this.camTheta),
    ]
    mat4Perspective(this.projMatrix, Math.PI / 3, this.canvas.width / this.canvas.height, 0.1, 1000)
    mat4LookAt(this.viewMatrix, eye, [0, 0, 0], [0, 1, 0])
    mat4Mul(this.vpMatrix, this.projMatrix, this.viewMatrix)
    this.sceneData.set(this.vpMatrix, 0)
    this.device.queue.writeBuffer(this.sceneBuffer, 0, this.sceneData)

    for (let i = 0; i < this.meshes.length; i++) {
      const mesh = this.meshes[i]
      mesh.rx += mesh.rxSpeed * dt
      mesh.ry += mesh.rySpeed * dt
      mesh.rz += mesh.rzSpeed * dt
      mesh.writeObjectData(this.objectStaging, i * this.objectFloatStride)
    }
    this.device.queue.writeBuffer(this.objectBuffer, 0, this.objectStaging.buffer, 0, this.meshes.length * this.objectStride)

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
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })

    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.sceneBindGroup)

    let currentGeometry: BufferGeometry | null = null
    for (let i = 0; i < this.meshes.length; i++) {
      const geometry = this.meshes[i].geometry
      if (geometry !== currentGeometry) {
        currentGeometry = geometry
        pass.setVertexBuffer(0, geometry.vertexBuffer)
        pass.setIndexBuffer(geometry.indexBuffer, 'uint16')
      }
      pass.setBindGroup(1, this.objectBindGroup, [i * this.objectStride])
      pass.drawIndexed(geometry.indexCount)
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

    for (const mesh of this.meshes) {
      this.destroyMesh(mesh)
    }
    this.boxGeometry?.destroy()
    this.boxGeometry = null
    this.meshes = []

    this.sceneBuffer?.destroy()
    this.objectBuffer?.destroy()
    this.depthTexture?.destroy()
    this.device?.destroy()
  }

  getInfo() {
    return 'Experiment A (WebGPU uniforms)'
  }
}
