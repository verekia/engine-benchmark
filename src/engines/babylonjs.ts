import type { EngineAdapter, BackendType, UseCase } from '../types'
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
  SceneLoader,
  AnimationGroup,
  TransformNode,
} from '@babylonjs/core'
import '@babylonjs/loaders/glTF'

interface SkinnedCharacter {
  root: TransformNode
  animationGroups: AnimationGroup[]
}

export class BabylonAdapter implements EngineAdapter {
  private engine!: Engine | WebGPUEngine
  private scene!: Scene
  private dirLight!: DirectionalLight
  private shadowGen: ShadowGenerator | null = null
  private cubes: InstanceType<typeof import('@babylonjs/core').Mesh>[] = []
  private characters: SkinnedCharacter[] = []
  private shadowsEnabled = false
  private canvas!: HTMLCanvasElement
  private useCase: UseCase = 'boxes'
  private baseContainer: any = null

  async init(canvas: HTMLCanvasElement, backend: BackendType, useCase: UseCase) {
    this.canvas = canvas
    this.useCase = useCase

    if (backend === 'webgpu') {
      const gpuEngine = new WebGPUEngine(canvas, { antialias: true })
      await gpuEngine.initAsync()
      this.engine = gpuEngine
    } else {
      this.engine = new Engine(canvas, true)
    }

    this.scene = new Scene(this.engine)
    this.scene.clearColor = new Color4(0, 0, 0, 1)

    if (useCase === 'boxes') {
      const camera = new ArcRotateCamera('camera', -Math.PI / 4, Math.PI / 3, 80, Vector3.Zero(), this.scene)
      camera.minZ = 0.1
      camera.attachControl(canvas, true)
    } else {
      const camera = new ArcRotateCamera('camera', -Math.PI / 4, Math.PI / 3, 50, new Vector3(0, 5, 0), this.scene)
      camera.minZ = 0.1
      camera.attachControl(canvas, true)
    }

    const ambient = new HemisphericLight('ambient', new Vector3(0, 1, 0), this.scene)
    ambient.intensity = 0.4

    this.dirLight = new DirectionalLight('dirLight', new Vector3(-1, -2, -1), this.scene)
    this.dirLight.position = new Vector3(30, 50, 30)
    this.dirLight.intensity = 1.5

    if (useCase === 'skinned-mesh') {
      // Load model as asset container for cloning
      this.baseContainer = await SceneLoader.LoadAssetContainerAsync(
        '/models/',
        'Michelle.glb',
        this.scene,
      )
    }

    window.addEventListener('resize', this.onResize)
  }

  private onResize = () => {
    this.engine.resize()
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
      mat.specularColor = Color3.Black()
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

  private setCharacterCount(count: number) {
    if (!this.baseContainer) return

    // Remove excess characters
    while (this.characters.length > count) {
      const char = this.characters.pop()!
      for (const ag of char.animationGroups) {
        ag.stop()
        ag.dispose()
      }
      char.root.dispose()
    }

    // Add missing characters
    const cols = Math.ceil(Math.sqrt(count))
    const spacing = 2
    const containerAnimGroups = this.baseContainer.animationGroups as AnimationGroup[]
    const totalAnimCount = containerAnimGroups.length

    while (this.characters.length < count) {
      const i = this.characters.length

      // Instantiate a new copy from the container
      const instance = this.baseContainer.instantiateModelsToScene(
        (name: string) => `${name}_${i}`,
        false,
      )

      const root = instance.rootNodes[0] as TransformNode
      const animGroups = instance.animationGroups as AnimationGroup[]

      // Grid layout
      const row = Math.floor(i / cols)
      const col = i % cols
      root.position = new Vector3(
        (col - cols / 2) * spacing,
        0,
        (row - cols / 2) * spacing,
      )
      root.scaling = new Vector3(0.01, 0.01, 0.01)

      // Alternate animations if multiple exist
      const animIndex = totalAnimCount > 1 ? i % totalAnimCount : 0

      // Stop all animation groups, then play the selected one
      for (const ag of animGroups) {
        ag.stop()
      }
      if (animGroups.length > animIndex) {
        const ag = animGroups[animIndex]
        ag.start(true)

        // Offset animation time
        const duration = ag.to - ag.from
        const timeOffset = (i / count) * duration
        ag.goToFrame(ag.from + timeOffset)
      }

      // Set shadow properties on meshes
      if (this.shadowsEnabled && this.shadowGen) {
        for (const mesh of root.getChildMeshes()) {
          this.shadowGen.addShadowCaster(mesh)
          mesh.receiveShadows = true
        }
      }

      this.characters.push({ root, animationGroups: animGroups })
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
      for (const char of this.characters) {
        for (const mesh of char.root.getChildMeshes()) {
          this.shadowGen.addShadowCaster(mesh)
          mesh.receiveShadows = true
        }
      }
    } else if (!enabled && this.shadowGen) {
      this.shadowGen.dispose()
      this.shadowGen = null
      for (const cube of this.cubes) {
        cube.receiveShadows = false
      }
      for (const char of this.characters) {
        for (const mesh of char.root.getChildMeshes()) {
          mesh.receiveShadows = false
        }
      }
    }
  }

  render(dt: number) {
    if (this.useCase === 'boxes') {
      const speed = 1.0
      for (const cube of this.cubes) {
        cube.rotation.x += speed * dt
        cube.rotation.y += speed * dt * 0.7
      }
    }
    // Babylon.js handles animation updates automatically via the scene
    this.scene.render()
  }

  dispose() {
    window.removeEventListener('resize', this.onResize)
    if (this.shadowGen) {
      this.shadowGen.dispose()
      this.shadowGen = null
    }
    for (const char of this.characters) {
      for (const ag of char.animationGroups) {
        ag.stop()
        ag.dispose()
      }
      char.root.dispose()
    }
    this.characters = []
    if (this.baseContainer) {
      this.baseContainer.dispose()
      this.baseContainer = null
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
