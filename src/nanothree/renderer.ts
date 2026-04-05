// High-performance WebGPU renderer for nanothree
//
// Key performance strategies vs Three.js (without instancing or batching):
// 1. Single storage buffer with dynamic offsets for all per-object data,
//    avoiding per-object bind groups or uniform buffer switches.
// 2. Interleaved vertex data (pos+normal) for cache locality.
// 3. Pre-allocated, growable staging buffer - zero per-frame allocations.
// 4. Inline model matrix computation - no intermediate objects.
// 5. Reused render pass descriptors and command encoder patterns.
// 6. One draw call per mesh - no tricks, just a fast core.

import type { PerspectiveCamera } from './core'
import type { Scene } from './scene'
import type { BufferGeometry } from './geometry'

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

// 20 floats per object: mat4x4 (16) + vec4 color (4)
const OBJECT_FLOATS = 20
const INITIAL_CAPACITY = 1024

export class WebGPURenderer {
  private device!: GPUDevice
  private context!: GPUCanvasContext
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
  private sceneLayout!: GPUBindGroupLayout
  private objectLayout!: GPUBindGroupLayout

  // Dynamic offset stride (aligned to device limits)
  private objectStride = 256
  private objectFloatStride = 64

  // Pre-allocated CPU-side staging
  private sceneData = new Float32Array(28) // viewProj(16) + lightDir(4) + ambient(4) + lightColor(4)
  private objectStaging!: Float32Array
  private capacity = INITIAL_CAPACITY

  // Render pass descriptors (reused every frame)
  private colorAtt: GPURenderPassColorAttachment
  private depthAtt: GPURenderPassDepthStencilAttachment
  private passDesc: GPURenderPassDescriptor

  shadowMap = { enabled: false }

  constructor(params: { canvas: HTMLCanvasElement; antialias?: boolean }) {
    this.canvas = params.canvas

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
  }

  get domElement() {
    return this.canvas
  }

  async init() {
    if (!navigator.gpu) throw new Error('WebGPU not supported')
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('No WebGPU adapter found')
    this.device = await adapter.requestDevice() as GPUDevice

    this.context = this.canvas.getContext('webgpu')!
    const format = navigator.gpu.getPreferredCanvasFormat()

    const dpr = window.devicePixelRatio
    this.canvas.width = (this.canvas.clientWidth * dpr) | 0
    this.canvas.height = (this.canvas.clientHeight * dpr) | 0

    this.context.configure({
      device: this.device,
      format,
      alphaMode: 'premultiplied',
    })

    // Compute aligned stride for dynamic offsets
    const align = this.device.limits.minStorageBufferOffsetAlignment
    this.objectStride = Math.ceil((OBJECT_FLOATS * 4) / align) * align
    this.objectFloatStride = this.objectStride / 4

    this.objectStaging = new Float32Array(INITIAL_CAPACITY * this.objectFloatStride)

    this.createPipeline(format)
    this.createBuffers(INITIAL_CAPACITY)
    this.createBindGroups()
    this.ensureDepthTexture()
  }

  private createPipeline(format: GPUTextureFormat) {
    const shader = this.device.createShaderModule({ code: SHADER })

    this.sceneLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })

    this.objectLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage', hasDynamicOffset: true } },
      ],
    })

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.sceneLayout, this.objectLayout],
      }),
      vertex: {
        module: shader,
        entryPoint: 'vs',
        buffers: [{
          arrayStride: 24, // 6 floats: pos(3) + normal(3)
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat },
            { shaderLocation: 1, offset: 12, format: 'float32x3' as GPUVertexFormat },
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
  }

  private createBuffers(capacity: number) {
    this.sceneBuffer = this.device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.objectBuffer = this.device.createBuffer({
      size: capacity * this.objectStride,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
  }

  private createBindGroups() {
    this.sceneBindGroup = this.device.createBindGroup({
      layout: this.sceneLayout,
      entries: [{ binding: 0, resource: { buffer: this.sceneBuffer } }],
    })
    this.objectBindGroup = this.device.createBindGroup({
      layout: this.objectLayout,
      entries: [{ binding: 0, resource: { buffer: this.objectBuffer, size: OBJECT_FLOATS * 4 } }],
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
    this.depthView = this.depthTexture.createView()
  }

  private grow(needed: number) {
    let newCap = this.capacity
    while (newCap < needed) newCap *= 2
    this.capacity = newCap
    this.objectStaging = new Float32Array(newCap * this.objectFloatStride)
    this.objectBuffer.destroy()
    this.objectBuffer = this.device.createBuffer({
      size: newCap * this.objectStride,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.objectBindGroup = this.device.createBindGroup({
      layout: this.objectLayout,
      entries: [{ binding: 0, resource: { buffer: this.objectBuffer, size: OBJECT_FLOATS * 4 } }],
    })
  }

  setSize(width: number, height: number, _updateStyle = true) {
    const dpr = window.devicePixelRatio
    this.canvas.width = (width * dpr) | 0
    this.canvas.height = (height * dpr) | 0
    this.ensureDepthTexture()
  }

  setPixelRatio(_ratio: number) {
    // Handled internally
  }

  render(scene: Scene, camera: PerspectiveCamera) {
    const meshes = scene.meshes
    const meshCount = meshes.length
    if (meshCount === 0) return

    // Handle resize
    const dpr = window.devicePixelRatio
    const w = (this.canvas.clientWidth * dpr) | 0
    const h = (this.canvas.clientHeight * dpr) | 0
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
      this.ensureDepthTexture()
    }

    // Ensure capacity
    if (meshCount > this.capacity) {
      this.grow(meshCount)
    }

    // Update camera
    camera.updateViewProjection(w / h)

    // Write scene uniforms
    this.sceneData.set(camera.viewProjection, 0)

    // Light direction (normalized)
    const dl = scene.directionalLights[0]
    if (dl) {
      const lx = dl.position.x, ly = dl.position.y, lz = dl.position.z
      const len = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1
      this.sceneData[16] = lx / len
      this.sceneData[17] = ly / len
      this.sceneData[18] = lz / len
    }

    // Ambient color
    const al = scene.ambientLights[0]
    if (al) {
      this.sceneData[20] = al.color.r * al.intensity
      this.sceneData[21] = al.color.g * al.intensity
      this.sceneData[22] = al.color.b * al.intensity
    }

    // Directional light color
    if (dl) {
      this.sceneData[24] = dl.color.r * dl.intensity
      this.sceneData[25] = dl.color.g * dl.intensity
      this.sceneData[26] = dl.color.b * dl.intensity
    }

    this.device.queue.writeBuffer(this.sceneBuffer, 0, this.sceneData)

    // Write per-object data into staging buffer
    const staging = this.objectStaging
    const floatStride = this.objectFloatStride

    for (let i = 0; i < meshCount; i++) {
      const mesh = meshes[i]
      const off = i * floatStride

      // Compute model matrix inline (Euler XYZ, with scale)
      const rx = mesh.rotation.x, ry = mesh.rotation.y, rz = mesh.rotation.z
      const cx = Math.cos(rx), sx = Math.sin(rx)
      const cy = Math.cos(ry), sy = Math.sin(ry)
      const cz = Math.cos(rz), sz = Math.sin(rz)
      const scx = mesh.scale.x, scy = mesh.scale.y, scz = mesh.scale.z

      staging[off]     = (cz * cy) * scx
      staging[off + 1] = (sz * cy) * scx
      staging[off + 2] = (-sy) * scx
      staging[off + 3] = 0
      staging[off + 4] = (cz * sy * sx - sz * cx) * scy
      staging[off + 5] = (sz * sy * sx + cz * cx) * scy
      staging[off + 6] = (cy * sx) * scy
      staging[off + 7] = 0
      staging[off + 8] = (cz * sy * cx + sz * sx) * scz
      staging[off + 9] = (sz * sy * cx - cz * sx) * scz
      staging[off + 10] = (cy * cx) * scz
      staging[off + 11] = 0
      staging[off + 12] = mesh.position.x
      staging[off + 13] = mesh.position.y
      staging[off + 14] = mesh.position.z
      staging[off + 15] = 1
      staging[off + 16] = mesh.material.color.r
      staging[off + 17] = mesh.material.color.g
      staging[off + 18] = mesh.material.color.b
      staging[off + 19] = 1
    }

    // Upload all object data in one call
    this.device.queue.writeBuffer(
      this.objectBuffer, 0,
      staging.buffer, 0,
      meshCount * this.objectStride,
    )

    // Begin render pass
    this.colorAtt.view = this.context.getCurrentTexture().createView()
    this.depthAtt.view = this.depthView

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(this.passDesc)

    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.sceneBindGroup)

    // One draw call per mesh, dynamic offset into the storage buffer
    let currentGeometry: BufferGeometry | null = null
    for (let i = 0; i < meshCount; i++) {
      const mesh = meshes[i]
      const geo = mesh.geometry

      // Only rebind vertex/index buffers when geometry changes
      if (geo !== currentGeometry) {
        currentGeometry = geo
        geo._ensureGPU(this.device)
        pass.setVertexBuffer(0, geo._vertexBuffer!)
        pass.setIndexBuffer(geo._indexBuffer!, 'uint16')
      }

      pass.setBindGroup(1, this.objectBindGroup, [i * this.objectStride])
      pass.drawIndexed(geo._indexCount)
    }

    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  dispose() {
    this.sceneBuffer?.destroy()
    this.objectBuffer?.destroy()
    this.depthTexture?.destroy()
    this.device?.destroy()
  }
}
