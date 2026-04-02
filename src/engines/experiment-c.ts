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
@group(1) @binding(0) var<storage, read> objectData: ObjectData;

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
     0.5, -0.5, -0.5,   0.5,  0.5, -0.5,   0.5,  0.5,  0.5,   0.5, -0.5,  0.5,
    -0.5, -0.5,  0.5,  -0.5,  0.5,  0.5,  -0.5,  0.5, -0.5,  -0.5, -0.5, -0.5,
    -0.5,  0.5,  0.5,   0.5,  0.5,  0.5,   0.5,  0.5, -0.5,  -0.5,  0.5, -0.5,
    -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5, -0.5,  0.5,  -0.5, -0.5,  0.5,
    -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,
     0.5, -0.5, -0.5,  -0.5, -0.5, -0.5,  -0.5,  0.5, -0.5,   0.5,  0.5, -0.5,
  ])

  const normals = new Float32Array([
     1, 0, 0,   1, 0, 0,   1, 0, 0,   1, 0, 0,
    -1, 0, 0,  -1, 0, 0,  -1, 0, 0,  -1, 0, 0,
     0, 1, 0,   0, 1, 0,   0, 1, 0,   0, 1, 0,
     0,-1, 0,   0,-1, 0,   0,-1, 0,   0,-1, 0,
     0, 0, 1,   0, 0, 1,   0, 0, 1,   0, 0, 1,
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
  out[0] = rx; out[1] = sx; out[2] = -fx; out[3] = 0
  out[4] = ry; out[5] = sy; out[6] = -fy; out[7] = 0
  out[8] = rz; out[9] = sz; out[10] = -fz; out[11] = 0
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
  color: [number, number, number]

  constructor(color: [number, number, number]) {
    this.color = color
  }
}

class Mesh {
  constructor(
    readonly geometry: BufferGeometry,
    readonly material: LambertMaterial,
  ) {}
}

class Node {
  x = 0; y = 0; z = 0
  rx = 0; ry = 0; rz = 0
  sx = 1; sy = 1; sz = 1
  mesh: Mesh | null = null

  writeObjectData(target: Float32Array, offset: number) {
    const mesh = this.mesh
    if (!mesh) return

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
    target[offset + 16] = mesh.material.color[0]
    target[offset + 17] = mesh.material.color[1]
    target[offset + 18] = mesh.material.color[2]
    target[offset + 19] = 1
  }
}

class Camera {
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

const INITIAL_CAPACITY = 1024
const OBJECT_FLOATS = 20

class Renderer {
  readonly device: GPUDevice
  readonly nodes: Node[] = []

  private context: GPUCanvasContext
  private canvas: HTMLCanvasElement
  private pipeline!: GPURenderPipeline
  private depthTexture!: GPUTexture
  private depthView!: GPUTextureView
  private depthW = 0
  private depthH = 0

  private sceneBuffer!: GPUBuffer
  private objectBuffer!: GPUBuffer
  private sceneBindGroup!: GPUBindGroup
  private objectBindGroup!: GPUBindGroup
  private objectLayout!: GPUBindGroupLayout

  private sceneData = new Float32Array(28)
  private objectStride = 256
  private objectFloatStride = 64
  private objectStaging = new Float32Array(INITIAL_CAPACITY * 64)
  private drawNodes: Node[] = []
  private capacity = INITIAL_CAPACITY

  private colorAtt: GPURenderPassColorAttachment
  private depthAtt: GPURenderPassDepthStencilAttachment
  private passDesc: GPURenderPassDescriptor

  private constructor(device: GPUDevice, context: GPUCanvasContext, canvas: HTMLCanvasElement) {
    this.device = device
    this.context = context
    this.canvas = canvas

    this.colorAtt = {
      view: undefined as unknown as GPUTextureView,
      clearValue: { r: 0.05, g: 0.05, b: 0.07, a: 1 },
      loadOp: 'clear',
      storeOp: 'store',
    }
    this.depthAtt = {
      view: undefined as unknown as GPUTextureView,
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    }
    this.passDesc = {
      colorAttachments: [this.colorAtt],
      depthStencilAttachment: this.depthAtt,
    }

    this.sceneData[16] = 0.5
    this.sceneData[17] = 1.0
    this.sceneData[18] = 0.3
    this.sceneData[20] = 0.18
    this.sceneData[21] = 0.18
    this.sceneData[22] = 0.2
    this.sceneData[24] = 1.0
    this.sceneData[25] = 0.95
    this.sceneData[26] = 0.9
  }

  static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
    if (!navigator.gpu) throw new Error('WebGPU not supported')
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('No WebGPU adapter found')
    const device = await adapter.requestDevice() as GPUDevice

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
    const align = this.device.limits.minStorageBufferOffsetAlignment
    this.objectStride = Math.ceil((OBJECT_FLOATS * 4) / align) * align
    this.objectFloatStride = this.objectStride / 4
    this.objectStaging = new Float32Array(this.capacity * this.objectFloatStride)

    this.sceneBuffer = this.device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.objectBuffer = this.device.createBuffer({
      size: this.capacity * this.objectStride,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    const sceneLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    })
    this.objectLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage', hasDynamicOffset: true } }],
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
    this.depthView = this.depthTexture.createView()
  }

  private grow(newCapacity: number) {
    this.capacity = newCapacity
    this.objectStaging = new Float32Array(newCapacity * this.objectFloatStride)
    this.objectBuffer.destroy()
    this.objectBuffer = this.device.createBuffer({
      size: newCapacity * this.objectStride,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.objectBindGroup = this.device.createBindGroup({
      layout: this.objectLayout,
      entries: [{ binding: 0, resource: { buffer: this.objectBuffer, size: OBJECT_FLOATS * 4 } }],
    })
  }

  createNode() {
    if (this.nodes.length >= this.capacity) {
      this.grow(this.capacity * 2)
    }
    const node = new Node()
    this.nodes.push(node)
    return node
  }

  removeNode(node: Node) {
    const index = this.nodes.indexOf(node)
    if (index === -1) return
    const last = this.nodes.length - 1
    if (index !== last) {
      this.nodes[index] = this.nodes[last]
    }
    this.nodes.pop()
  }

  handleResize() {
    const dpr = devicePixelRatio
    const w = (this.canvas.clientWidth * dpr) | 0
    const h = (this.canvas.clientHeight * dpr) | 0
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
      this.ensureDepthTexture()
    }
  }

  render(camera: Camera) {
    if (this.nodes.length === 0) return

    this.handleResize()
    camera.update(this.canvas.width / this.canvas.height)
    this.sceneData.set(camera.viewProjection, 0)
    this.device.queue.writeBuffer(this.sceneBuffer, 0, this.sceneData)

    let drawCount = 0
    for (let i = 0; i < this.nodes.length; i++) {
      const node = this.nodes[i]
      if (!node.mesh) continue
      node.writeObjectData(this.objectStaging, drawCount * this.objectFloatStride)
      this.drawNodes[drawCount] = node
      drawCount++
    }

    if (drawCount === 0) return

    this.device.queue.writeBuffer(this.objectBuffer, 0, this.objectStaging.buffer, 0, drawCount * this.objectStride)

    this.colorAtt.view = this.context.getCurrentTexture().createView()
    this.depthAtt.view = this.depthView

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(this.passDesc)

    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.sceneBindGroup)

    let currentGeometry: BufferGeometry | null = null
    for (let drawIndex = 0; drawIndex < drawCount; drawIndex++) {
      const mesh = this.drawNodes[drawIndex].mesh!
      const geometry = mesh.geometry
      if (geometry !== currentGeometry) {
        currentGeometry = geometry
        pass.setVertexBuffer(0, geometry.vertexBuffer)
        pass.setIndexBuffer(geometry.indexBuffer, 'uint16')
      }
      pass.setBindGroup(1, this.objectBindGroup, [drawIndex * this.objectStride])
      pass.drawIndexed(geometry.indexCount)
    }

    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  dispose() {
    this.sceneBuffer?.destroy()
    this.objectBuffer?.destroy()
    this.depthTexture?.destroy()
    this.device?.destroy()
    this.nodes.length = 0
    this.drawNodes.length = 0
  }
}

interface AnimatedNode {
  node: Node
  mesh: Mesh
  geometry: BufferGeometry | null
  rxSpeed: number
  rySpeed: number
  rzSpeed: number
}

export class ExperimentCAdapter implements EngineAdapter {
  private renderer!: Renderer
  private camera = new Camera()
  private animatedNodes: AnimatedNode[] = []
  private boxGeometry: BufferGeometry | null = null
  private canvas!: HTMLCanvasElement
  private useCase: UseCase = 'boxes'

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

    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerup', this.onPointerUp)
    canvas.addEventListener('wheel', this.onWheel, { passive: true })
  }

  private destroyAnimatedNode(entry: AnimatedNode) {
    this.renderer.removeNode(entry.node)
    if (entry.geometry && entry.geometry !== this.boxGeometry) {
      entry.geometry.destroy()
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
    this.camTheta -= (e.clientX - this.lastMX) * 0.005
    this.camPhi = Math.max(0.05, Math.min(Math.PI - 0.05, this.camPhi + (e.clientY - this.lastMY) * 0.005))
    this.lastMX = e.clientX
    this.lastMY = e.clientY
  }

  private onPointerUp = () => {
    this.dragging = false
  }

  private onWheel = (e: WheelEvent) => {
    this.camDist = Math.max(5, Math.min(300, this.camDist + e.deltaY * 0.05))
  }

  setMeshCount(count: number) {
    if (this.useCase !== 'boxes' && this.useCase !== 'unique-tetrahedra') return

    while (this.animatedNodes.length > count) {
      this.destroyAnimatedNode(this.animatedNodes.pop()!)
    }

    if (this.useCase === 'boxes') {
      if (!this.boxGeometry) {
        this.boxGeometry = new BufferGeometry(this.renderer.device, createBoxGeometryData())
      }

      const spread = 50
      while (this.animatedNodes.length < count) {
        const node = this.renderer.createNode()
        const mesh = new Mesh(
          this.boxGeometry,
          new LambertMaterial([Math.random(), Math.random(), Math.random()]),
        )
        node.mesh = mesh
        node.x = (Math.random() - 0.5) * spread
        node.y = (Math.random() - 0.5) * spread
        node.z = (Math.random() - 0.5) * spread

        this.animatedNodes.push({
          node,
          mesh,
          geometry: null,
          rxSpeed: 1.0,
          rySpeed: 0.7,
          rzSpeed: 0,
        })
      }
    } else {
      while (this.animatedNodes.length < count) {
        const spec = createUniqueTetrahedronSpec(this.animatedNodes.length)
        const geometry = new BufferGeometry(this.renderer.device, spec.geometry)
        const mesh = new Mesh(geometry, new LambertMaterial(spec.material.color))
        const node = this.renderer.createNode()
        node.mesh = mesh
        node.x = spec.position[0]
        node.y = spec.position[1]
        node.z = spec.position[2]
        node.rx = spec.rotation[0]
        node.ry = spec.rotation[1]
        node.rz = spec.rotation[2]

        this.animatedNodes.push({
          node,
          mesh,
          geometry,
          rxSpeed: spec.rotationSpeed[0],
          rySpeed: spec.rotationSpeed[1],
          rzSpeed: spec.rotationSpeed[2],
        })
      }
    }
  }

  setShadows(_enabled: boolean) {
    // Shadows not implemented in this minimal experiment.
  }

  render(dt: number) {
    if ((this.useCase !== 'boxes' && this.useCase !== 'unique-tetrahedra') || this.animatedNodes.length === 0) return

    const sinPhi = Math.sin(this.camPhi)
    this.camera.position[0] = this.camDist * sinPhi * Math.sin(this.camTheta)
    this.camera.position[1] = this.camDist * Math.cos(this.camPhi)
    this.camera.position[2] = this.camDist * sinPhi * Math.cos(this.camTheta)

    for (let i = 0; i < this.animatedNodes.length; i++) {
      const entry = this.animatedNodes[i]
      entry.node.rx += entry.rxSpeed * dt
      entry.node.ry += entry.rySpeed * dt
      entry.node.rz += entry.rzSpeed * dt
    }

    this.renderer.render(this.camera)
  }

  dispose() {
    this.canvas.removeEventListener('pointerdown', this.onPointerDown)
    this.canvas.removeEventListener('pointermove', this.onPointerMove)
    this.canvas.removeEventListener('pointerup', this.onPointerUp)
    this.canvas.removeEventListener('wheel', this.onWheel)

    for (const entry of this.animatedNodes) {
      this.destroyAnimatedNode(entry)
    }
    this.boxGeometry?.destroy()
    this.boxGeometry = null
    this.animatedNodes = []
    this.renderer?.dispose()
  }

  getInfo() {
    return 'Experiment C (WebGPU engine)'
  }
}
