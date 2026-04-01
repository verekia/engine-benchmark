import type { EngineAdapter, BackendType } from '../types'
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
} from 'voidcore'
import type { Quat, Vec3 } from 'voidcore'

export class VoidcoreAdapter implements EngineAdapter {
  private engine!: VoidEngine
  private scene!: Scene
  private camera!: PerspectiveCamera
  private cubes: Mesh[] = []
  private dirLight!: DirectionalLight
  private geometry!: BoxGeometry
  private shadowsEnabled = false
  private unregisterUpdate?: () => void
  private unregisterRender?: () => void
  private controls!: OrbitControls

  async init(canvas: HTMLCanvasElement, backend: BackendType) {
    this.engine = await VoidEngine.create(canvas, {
      backend: backend === 'webgl' ? 'webgl2' : 'webgpu',
      antialias: true,
      shadows: true,
    })

    this.scene = new Scene()
    this.camera = new PerspectiveCamera({ fov: 60, far: 1000 })
    this.camera.setPosition(0, -60, 40)

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

    this.geometry = new BoxGeometry()

    const tempQuat = quatCreate()
    const axisX: Vec3 = new Float32Array([1, 0, 0]) as any
    const axisZ: Vec3 = new Float32Array([0, 0, 1]) as any

    this.unregisterUpdate = this.engine.register(
      ({ dt }) => {
        this.controls.update(dt)
        const speed = 1.0
        for (const cube of this.cubes) {
          // Rotate around X axis
          quatFromAxisAngle(tempQuat, axisX, speed * dt)
          quatMultiply(cube.rotation, tempQuat, cube.rotation)
          // Rotate around Z axis
          quatFromAxisAngle(tempQuat, axisZ, speed * dt * 0.7)
          quatMultiply(cube.rotation, tempQuat, cube.rotation)
          cube.markTransformDirty()
        }
      },
      { priority: -1 },
    )

    this.unregisterRender = this.engine.register(
      () => {
        this.engine.render(this.scene, this.camera)
      },
      { priority: 0 },
    )

    this.engine.maxFps = 0 // uncapped
    this.engine.maxDpr = false as any
    this.engine.start()
  }

  setCubeCount(count: number) {
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

  setShadows(enabled: boolean) {
    this.shadowsEnabled = enabled
    this.dirLight.castShadow = enabled
    for (const cube of this.cubes) {
      cube.castShadow = enabled
      cube.receiveShadow = enabled
    }
  }

  render(_dt: number) {
    // Voidcore handles its own render loop via engine.register
  }

  dispose() {
    this.unregisterUpdate?.()
    this.unregisterRender?.()
    this.engine.dispose()
    this.cubes = []
  }

  getInfo(): string {
    return `Voidcore (${this.engine.backend})`
  }
}
