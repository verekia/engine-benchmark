import type { EngineAdapter, BackendType, UseCase } from '../types'
import { createUniqueTetrahedronSpec } from '../benchmark-scene'
import {
  Engine as VoidEngine,
  Scene,
  PerspectiveCamera,
  Mesh,
  BoxGeometry,
  LambertMaterial,
  AmbientLight,
  DirectionalLight,
  OrbitControls,
  quatFromAxisAngle,
  quatMultiply,
  quatCreate,
  loadGLTF,
  cloneScene,
  AnimationMixer,
  AnimationClip,
  Geometry,
} from 'voidcore'
import type { Vec3, Quat } from 'voidcore'

interface SkinnedCharacter {
  model: any
  mixer: InstanceType<typeof AnimationMixer>
}

interface AnimatedStaticMesh {
  mesh: Mesh
  geometry: Geometry | null
  material: LambertMaterial
  rotationSpeed: [number, number, number]
}

export class VoidcoreAdapter implements EngineAdapter {
  private engine!: VoidEngine
  private scene!: Scene
  private camera!: PerspectiveCamera
  private boxes: AnimatedStaticMesh[] = []
  private tetrahedra: AnimatedStaticMesh[] = []
  private characters: SkinnedCharacter[] = []
  private dirLight!: DirectionalLight
  private boxGeometry: BoxGeometry | null = null
  private shadowsEnabled = false
  private unregisterUpdate?: () => void
  private unregisterRender?: () => void
  private controls!: OrbitControls
  private useCase: UseCase = 'boxes'
  private baseGLTF: any = null

  async init(canvas: HTMLCanvasElement, backend: BackendType, useCase: UseCase) {
    this.useCase = useCase

    this.engine = await VoidEngine.create(canvas, {
      backend: backend === 'webgl' ? 'webgl2' : 'webgpu',
      antialias: true,
      shadows: true,
    })

    this.scene = new Scene()
    this.camera = new PerspectiveCamera({ fov: 60, far: 1000 })

    if (useCase === 'skinned-mesh') {
      this.camera.setPosition(0, -40, 15)
    } else {
      this.camera.setPosition(0, -60, 40)
    }

    this.controls = new OrbitControls(this.camera, canvas)

    const ambient = new AmbientLight({ color: [0.25, 0.25, 0.25], intensity: 1 })
    this.scene.add(ambient)

    this.dirLight = new DirectionalLight({
      intensity: 1.5,
      castShadow: false,
      shadowMapSize: 120,
    })
    this.dirLight.setPosition(30, -30, 50)
    this.scene.add(this.dirLight)

    if (useCase === 'skinned-mesh') {
      this.baseGLTF = await loadGLTF('/models/Michelle.glb')
      this.unregisterUpdate = this.engine.register(
        ({ dt }) => {
          this.controls.update(dt)
          for (const char of this.characters) {
            char.mixer.update(dt)
          }
        },
        { priority: -1 },
      )
    } else {
      this.boxGeometry = new BoxGeometry()
      const tempQuat = quatCreate()
      const axisX: Vec3 = new Float32Array([1, 0, 0]) as any
      const axisY: Vec3 = new Float32Array([0, 1, 0]) as any
      const axisZ: Vec3 = new Float32Array([0, 0, 1]) as any

      this.unregisterUpdate = this.engine.register(
        ({ dt }) => {
          this.controls.update(dt)
          const staticMeshes = this.useCase === 'boxes' ? this.boxes : this.tetrahedra
          for (const entry of staticMeshes) {
            this.rotateQuat(entry.mesh.rotation, axisX, entry.rotationSpeed[0] * dt, tempQuat)
            this.rotateQuat(entry.mesh.rotation, axisY, entry.rotationSpeed[1] * dt, tempQuat)
            this.rotateQuat(entry.mesh.rotation, axisZ, entry.rotationSpeed[2] * dt, tempQuat)
            entry.mesh.markTransformDirty()
          }
        },
        { priority: -1 },
      )
    }

    this.unregisterRender = this.engine.register(
      () => {
        this.engine.render(this.scene, this.camera)
      },
      { priority: 0 },
    )

    this.engine.maxFps = 0
    this.engine.maxDpr = Infinity
    this.engine.start()
  }

  private rotateQuat(target: Quat, axis: Vec3, angle: number, tempQuat: Quat) {
    quatFromAxisAngle(tempQuat, axis, angle)
    quatMultiply(target, tempQuat, target)
  }

  private setInitialEulerRotation(mesh: Mesh, rx: number, ry: number, rz: number) {
    const qx = quatCreate()
    const qy = quatCreate()
    const qz = quatCreate()
    const axisX: Vec3 = new Float32Array([1, 0, 0]) as any
    const axisY: Vec3 = new Float32Array([0, 1, 0]) as any
    const axisZ: Vec3 = new Float32Array([0, 0, 1]) as any

    quatFromAxisAngle(qx, axisX, rx)
    quatFromAxisAngle(qy, axisY, ry)
    quatFromAxisAngle(qz, axisZ, rz)
    quatMultiply(mesh.rotation, qy, qx)
    quatMultiply(mesh.rotation, qz, mesh.rotation)
    mesh.markTransformDirty()
  }

  private destroyStaticMesh(entry: AnimatedStaticMesh) {
    this.scene.remove(entry.mesh)
    entry.geometry?.dispose()
  }

  setMeshCount(count: number) {
    if (this.useCase === 'boxes') {
      this.setBoxCount(count)
    } else if (this.useCase === 'unique-tetrahedra') {
      this.setUniqueTetrahedronCount(count)
    } else {
      this.setCharacterCount(count)
    }
  }

  private setBoxCount(count: number) {
    while (this.boxes.length > count) {
      this.destroyStaticMesh(this.boxes.pop()!)
    }

    const spread = 50
    while (this.boxes.length < count) {
      const r = Math.random()
      const g = Math.random()
      const b = Math.random()
      const material = new LambertMaterial({ color: [r, g, b] })
      const mesh = new Mesh(this.boxGeometry!, material)
      mesh.setPosition(
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
      )
      mesh.castShadow = this.shadowsEnabled
      mesh.receiveShadow = this.shadowsEnabled
      this.scene.add(mesh)
      this.boxes.push({
        mesh,
        geometry: null,
        material,
        rotationSpeed: [1.0, 0.7, 0],
      })
    }
  }

  private setUniqueTetrahedronCount(count: number) {
    while (this.tetrahedra.length > count) {
      this.destroyStaticMesh(this.tetrahedra.pop()!)
    }

    while (this.tetrahedra.length < count) {
      const spec = createUniqueTetrahedronSpec(this.tetrahedra.length)
      const geometry = new Geometry(spec.geometry)
      const material = new LambertMaterial({ color: spec.material.color })
      const mesh = new Mesh(geometry, material)
      mesh.setPosition(...spec.position)
      this.setInitialEulerRotation(mesh, ...spec.rotation)
      mesh.castShadow = this.shadowsEnabled
      mesh.receiveShadow = this.shadowsEnabled
      this.scene.add(mesh)
      this.tetrahedra.push({
        mesh,
        geometry,
        material,
        rotationSpeed: spec.rotationSpeed,
      })
    }
  }

  private setCharacterCount(count: number) {
    if (!this.baseGLTF) return

    const animations = this.baseGLTF.animations as AnimationClip[]

    while (this.characters.length > count) {
      const char = this.characters.pop()!
      char.mixer.stopAll()
      this.scene.remove(char.model)
    }

    const cols = Math.ceil(Math.sqrt(count))
    const spacing = 2

    while (this.characters.length < count) {
      const i = this.characters.length
      const cloneResult = cloneScene(this.baseGLTF.scene, this.baseGLTF.skeletons)
      const model = cloneResult.root

      const row = Math.floor(i / cols)
      const col = i % cols
      model.setPosition(
        (col - cols / 2) * spacing,
        (row - cols / 2) * spacing,
        0,
      )
      model.setScale(0.01, 0.01, 0.01)

      model.castShadow = this.shadowsEnabled
      model.receiveShadow = this.shadowsEnabled
      this.scene.add(model)

      const skeleton = cloneResult.skeletons.length > 0 ? cloneResult.skeletons[0] : null
      const mixer = new AnimationMixer(skeleton ?? (model as any))

      if (animations && animations.length > 0) {
        const clipIndex = animations.length > 1 ? i % animations.length : 0
        const action = mixer.clipAction(animations[clipIndex])
        action.play()

        const duration = animations[clipIndex].duration
        const timeOffset = (i / count) * duration
        mixer.update(timeOffset)
      }

      this.characters.push({ model, mixer })
    }
  }

  setShadows(enabled: boolean) {
    this.shadowsEnabled = enabled
    this.dirLight.castShadow = enabled
    for (const box of this.boxes) {
      box.mesh.castShadow = enabled
      box.mesh.receiveShadow = enabled
    }
    for (const tetra of this.tetrahedra) {
      tetra.mesh.castShadow = enabled
      tetra.mesh.receiveShadow = enabled
    }
    for (const char of this.characters) {
      char.model.castShadow = enabled
      char.model.receiveShadow = enabled
    }
  }

  render(_dt: number) {
    // Voidcore handles its own render loop via engine.register
  }

  dispose() {
    this.unregisterUpdate?.()
    this.unregisterRender?.()

    for (const box of this.boxes) {
      this.destroyStaticMesh(box)
    }
    for (const tetra of this.tetrahedra) {
      this.destroyStaticMesh(tetra)
    }
    for (const char of this.characters) {
      char.mixer.stopAll()
      this.scene.remove(char.model)
    }

    this.boxes = []
    this.tetrahedra = []
    this.characters = []
    this.boxGeometry = null
    if (this.baseGLTF?.dispose) {
      this.baseGLTF.dispose()
    }
    this.baseGLTF = null
    this.engine.dispose()
  }

  getInfo(): string {
    return `Voidcore (${this.engine.backend})`
  }
}
