import type { EngineAdapter, BackendType, UseCase } from '../types'
import { createUniqueTetrahedronSpec, type BufferGeometryData } from '../benchmark-scene'
import {
  Scene,
  PerspectiveCamera,
  BoxGeometry,
  BufferGeometry,
  Float32BufferAttribute,
  MeshLambertMaterial,
  Mesh,
  AmbientLight,
  DirectionalLight,
  Color,
  WebGPURenderer,
} from '../nanothree'

interface AnimatedStaticMesh {
  mesh: Mesh
  material: MeshLambertMaterial
  geometry: BufferGeometry
  rotationSpeed: [number, number, number]
}

export class NanoThreeAdapter implements EngineAdapter {
  private renderer!: WebGPURenderer
  private scene!: Scene
  private camera!: PerspectiveCamera
  private boxes: AnimatedStaticMesh[] = []
  private tetrahedra: AnimatedStaticMesh[] = []
  private dirLight!: DirectionalLight
  private boxGeometry: BoxGeometry | null = null
  private shadowsEnabled = false
  private useCase: UseCase = 'boxes'
  private canvas!: HTMLCanvasElement

  // Orbit camera state
  private camTheta = 0
  private camPhi = Math.acos(40 / 72)
  private camDist = 72
  private dragging = false
  private lastMX = 0
  private lastMY = 0

  async init(canvas: HTMLCanvasElement, backend: BackendType, useCase: UseCase) {
    if (backend !== 'webgpu') throw new Error('nanothree supports WebGPU only')

    this.canvas = canvas
    this.useCase = useCase
    this.scene = new Scene()
    this.camera = new PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 1000)
    this.camera.position.set(0, 40, 60)

    this.renderer = new WebGPURenderer({ canvas, antialias: true })
    await this.renderer.init()

    // Lights
    const ambient = new AmbientLight(0x404040, 1)
    this.scene.add(ambient)

    this.dirLight = new DirectionalLight(0xffffff, 1.5)
    this.dirLight.position.set(30, 50, 30)
    this.scene.add(this.dirLight)

    if (useCase === 'boxes') {
      this.boxGeometry = new BoxGeometry(1, 1, 1)
    }

    // Orbit controls
    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerup', this.onPointerUp)
    canvas.addEventListener('wheel', this.onWheel, { passive: true })
  }

  private createBufferGeometry(data: BufferGeometryData): BufferGeometry {
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new Float32BufferAttribute(data.positions, 3))
    geometry.setAttribute('normal', new Float32BufferAttribute(data.normals, 3))
    geometry.setIndex(Array.from(data.indices))
    return geometry
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
    if (this.useCase === 'boxes') {
      this.setBoxCount(count)
    } else if (this.useCase === 'unique-tetrahedra') {
      this.setUniqueTetrahedronCount(count)
    }
  }

  private setBoxCount(count: number) {
    while (this.boxes.length > count) {
      const box = this.boxes.pop()!
      this.scene.remove(box.mesh)
      box.material.dispose()
    }

    const spread = 50
    while (this.boxes.length < count) {
      const color = new Color(Math.random(), Math.random(), Math.random())
      const material = new MeshLambertMaterial({ color })
      const mesh = new Mesh(this.boxGeometry!, material)
      mesh.position.set(
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
      )
      mesh.castShadow = this.shadowsEnabled
      mesh.receiveShadow = this.shadowsEnabled
      this.scene.add(mesh)

      this.boxes.push({
        mesh,
        material,
        geometry: this.boxGeometry!,
        rotationSpeed: [1.0, 0.7, 0],
      })
    }
  }

  private setUniqueTetrahedronCount(count: number) {
    while (this.tetrahedra.length > count) {
      const tetra = this.tetrahedra.pop()!
      this.scene.remove(tetra.mesh)
      tetra.material.dispose()
      tetra.geometry.dispose()
    }

    while (this.tetrahedra.length < count) {
      const spec = createUniqueTetrahedronSpec(this.tetrahedra.length)
      const geometry = this.createBufferGeometry(spec.geometry)
      const material = new MeshLambertMaterial({
        color: new Color(...spec.material.color),
      })
      const mesh = new Mesh(geometry, material)
      mesh.position.set(...spec.position)
      mesh.rotation.set(...spec.rotation)
      mesh.castShadow = this.shadowsEnabled
      mesh.receiveShadow = this.shadowsEnabled
      this.scene.add(mesh)

      this.tetrahedra.push({
        mesh,
        material,
        geometry,
        rotationSpeed: spec.rotationSpeed,
      })
    }
  }

  setShadows(enabled: boolean) {
    this.shadowsEnabled = enabled
    this.renderer.shadowMap.enabled = enabled
    this.dirLight.castShadow = enabled

    for (const box of this.boxes) {
      box.mesh.castShadow = enabled
      box.mesh.receiveShadow = enabled
    }
    for (const tetra of this.tetrahedra) {
      tetra.mesh.castShadow = enabled
      tetra.mesh.receiveShadow = enabled
    }
  }

  render(dt: number) {
    const meshes = this.useCase === 'boxes' ? this.boxes : this.tetrahedra
    if (meshes.length === 0) return

    // Update orbit camera
    const sinPhi = Math.sin(this.camPhi)
    this.camera.position.set(
      this.camDist * sinPhi * Math.sin(this.camTheta),
      this.camDist * Math.cos(this.camPhi),
      this.camDist * sinPhi * Math.cos(this.camTheta),
    )

    // Animate rotations
    for (let i = 0; i < meshes.length; i++) {
      const entry = meshes[i]
      entry.mesh.rotation.x += entry.rotationSpeed[0] * dt
      entry.mesh.rotation.y += entry.rotationSpeed[1] * dt
      entry.mesh.rotation.z += entry.rotationSpeed[2] * dt
    }

    this.renderer.render(this.scene, this.camera)
  }

  dispose() {
    this.canvas.removeEventListener('pointerdown', this.onPointerDown)
    this.canvas.removeEventListener('pointermove', this.onPointerMove)
    this.canvas.removeEventListener('pointerup', this.onPointerUp)
    this.canvas.removeEventListener('wheel', this.onWheel)

    for (const box of this.boxes) {
      this.scene.remove(box.mesh)
      box.material.dispose()
    }
    for (const tetra of this.tetrahedra) {
      this.scene.remove(tetra.mesh)
      tetra.material.dispose()
      tetra.geometry.dispose()
    }

    this.boxes = []
    this.tetrahedra = []
    this.boxGeometry?.dispose()
    this.boxGeometry = null
    this.renderer.dispose()
  }

  getInfo(): string {
    return 'nanothree (WebGPU)'
  }
}
