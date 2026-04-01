import type { EngineAdapter, BackendType, UseCase } from '../types'
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
} from 'voidcore'
import type { Vec3 } from 'voidcore'

interface SkinnedCharacter {
  model: any
  mixer: InstanceType<typeof AnimationMixer>
}

export class VoidcoreAdapter implements EngineAdapter {
  private engine!: VoidEngine
  private scene!: Scene
  private camera!: PerspectiveCamera
  private cubes: Mesh[] = []
  private characters: SkinnedCharacter[] = []
  private dirLight!: DirectionalLight
  private geometry!: BoxGeometry
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

    if (useCase === 'boxes') {
      this.camera.setPosition(0, -60, 40)
    } else {
      this.camera.setPosition(0, -40, 15)
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

    if (useCase === 'boxes') {
      this.geometry = new BoxGeometry()

      const tempQuat = quatCreate()
      const axisX: Vec3 = new Float32Array([1, 0, 0]) as any
      const axisZ: Vec3 = new Float32Array([0, 0, 1]) as any

      this.unregisterUpdate = this.engine.register(
        ({ dt }) => {
          this.controls.update(dt)
          const speed = 1.0
          for (const cube of this.cubes) {
            quatFromAxisAngle(tempQuat, axisX, speed * dt)
            quatMultiply(cube.rotation, tempQuat, cube.rotation)
            quatFromAxisAngle(tempQuat, axisZ, speed * dt * 0.7)
            quatMultiply(cube.rotation, tempQuat, cube.rotation)
            cube.markTransformDirty()
          }
        },
        { priority: -1 },
      )
    } else {
      // Load the skinned mesh model
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
    }

    this.unregisterRender = this.engine.register(
      () => {
        this.engine.render(this.scene, this.camera)
      },
      { priority: 0 },
    )

    this.engine.maxFps = 0 // uncapped
    this.engine.maxDpr = Infinity // uncapped
    this.engine.start()
  }

  setMeshCount(count: number) {
    if (this.useCase === 'boxes') {
      this.setBoxCount(count)
    } else {
      this.setCharacterCount(count)
    }
  }

  private setBoxCount(count: number) {
    while (this.cubes.length > count) {
      const cube = this.cubes.pop()!
      this.scene.remove(cube)
    }

    const spread = 50
    while (this.cubes.length < count) {
      const r = Math.random()
      const g = Math.random()
      const b = Math.random()
      const mat = new LambertMaterial({ color: [r, g, b] })
      const mesh = new Mesh(this.geometry, mat)
      mesh.setPosition(
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
      )
      mesh.castShadow = this.shadowsEnabled
      mesh.receiveShadow = this.shadowsEnabled
      this.scene.add(mesh)
      this.cubes.push(mesh)
    }
  }

  private setCharacterCount(count: number) {
    if (!this.baseGLTF) return

    const animations = this.baseGLTF.animations as AnimationClip[]

    // Remove excess characters
    while (this.characters.length > count) {
      const char = this.characters.pop()!
      char.mixer.stopAll()
      this.scene.remove(char.model)
    }

    // Add missing characters
    const cols = Math.ceil(Math.sqrt(count))
    const spacing = 2

    while (this.characters.length < count) {
      const i = this.characters.length

      // Clone the scene hierarchy with skeletons
      const cloneResult = cloneScene(this.baseGLTF.scene, this.baseGLTF.skeletons)
      const model = cloneResult.root

      // Grid layout
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

      // Set up animation using cloned skeleton
      const skeleton = cloneResult.skeletons.length > 0 ? cloneResult.skeletons[0] : null
      const mixer = new AnimationMixer(skeleton ?? (model as any))

      if (animations && animations.length > 0) {
        const clipIndex = animations.length > 1 ? i % animations.length : 0
        const action = mixer.clipAction(animations[clipIndex])
        action.play()

        // Offset animation time
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
    for (const cube of this.cubes) {
      cube.castShadow = enabled
      cube.receiveShadow = enabled
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
    for (const char of this.characters) {
      char.mixer.stopAll()
      this.scene.remove(char.model)
    }
    this.characters = []
    if (this.baseGLTF?.dispose) {
      this.baseGLTF.dispose()
    }
    this.baseGLTF = null
    this.engine.dispose()
    this.cubes = []
  }

  getInfo(): string {
    return `Voidcore (${this.engine.backend})`
  }
}
