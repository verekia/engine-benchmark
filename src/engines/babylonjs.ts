import type { EngineAdapter, BackendType } from '../types'
import {
  Engine,
  WebGPUEngine,
  Scene,
  ArcRotateCamera,
  Vector3,
  HemisphericLight,
  DirectionalLight,
  ShadowGenerator,
  MeshBuilder,
  StandardMaterial,
  Color3,
  Color4,
} from '@babylonjs/core'

export class BabylonAdapter implements EngineAdapter {
  private engine!: Engine | WebGPUEngine
  private scene!: Scene
  private dirLight!: DirectionalLight
  private shadowGen: ShadowGenerator | null = null
  private cubes: InstanceType<typeof import('@babylonjs/core').Mesh>[] = []
  private shadowsEnabled = false
  private canvas!: HTMLCanvasElement

  async init(canvas: HTMLCanvasElement, backend: BackendType) {
    this.canvas = canvas

    if (backend === 'webgpu') {
      const gpuEngine = new WebGPUEngine(canvas, { antialias: true })
      await gpuEngine.initAsync()
      this.engine = gpuEngine
    } else {
      this.engine = new Engine(canvas, true)
    }

    this.scene = new Scene(this.engine)
    this.scene.clearColor = new Color4(0, 0, 0, 1)

    const camera = new ArcRotateCamera('camera', -Math.PI / 4, Math.PI / 3, 80, Vector3.Zero(), this.scene)
    camera.minZ = 0.1
    camera.attachControl(canvas, true)

    const ambient = new HemisphericLight('ambient', new Vector3(0, 1, 0), this.scene)
    ambient.intensity = 0.4

    this.dirLight = new DirectionalLight('dirLight', new Vector3(-1, -2, -1), this.scene)
    this.dirLight.position = new Vector3(30, 50, 30)
    this.dirLight.intensity = 1.5

    window.addEventListener('resize', this.onResize)
  }

  private onResize = () => {
    this.engine.resize()
  }

  setCubeCount(count: number) {
    while (this.cubes.length > count) {
      const cube = this.cubes.pop()!
      if (this.shadowGen) {
        this.shadowGen.removeShadowCaster(cube)
      }
      cube.dispose()
    }

    const spread = 50
    while (this.cubes.length < count) {
      const box = MeshBuilder.CreateBox(`cube_${this.cubes.length}`, { size: 1 }, this.scene)
      const mat = new StandardMaterial(`mat_${this.cubes.length}`, this.scene)
      mat.diffuseColor = new Color3(Math.random(), Math.random(), Math.random())
      box.material = mat
      box.position = new Vector3(
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
      )

      if (this.shadowsEnabled && this.shadowGen) {
        this.shadowGen.addShadowCaster(box)
        box.receiveShadows = true
      }

      this.cubes.push(box)
    }
  }

  setShadows(enabled: boolean) {
    this.shadowsEnabled = enabled

    if (enabled && !this.shadowGen) {
      this.shadowGen = new ShadowGenerator(2048, this.dirLight)
      this.shadowGen.useBlurExponentialShadowMap = true
      for (const cube of this.cubes) {
        this.shadowGen.addShadowCaster(cube)
        cube.receiveShadows = true
      }
    } else if (!enabled && this.shadowGen) {
      this.shadowGen.dispose()
      this.shadowGen = null
      for (const cube of this.cubes) {
        cube.receiveShadows = false
      }
    }
  }

  render(dt: number) {
    const speed = 1.0
    for (const cube of this.cubes) {
      cube.rotation.x += speed * dt
      cube.rotation.y += speed * dt * 0.7
    }
    this.scene.render()
  }

  dispose() {
    window.removeEventListener('resize', this.onResize)
    if (this.shadowGen) {
      this.shadowGen.dispose()
      this.shadowGen = null
    }
    this.scene.dispose()
    this.engine.dispose()
    this.cubes = []
  }

  getInfo(): string {
    const type = this.engine instanceof WebGPUEngine ? 'WebGPU' : 'WebGL'
    return `Babylon.js ${type}`
  }
}
