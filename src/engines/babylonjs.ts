import type { EngineAdapter, BackendType, UseCase } from '../types'
import { createUniqueTetrahedronSpec, type BufferGeometryData } from '../benchmark-scene'
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
  Mesh as BabylonMesh,
  VertexData,
} from '@babylonjs/core'
import '@babylonjs/loaders/glTF'

interface SkinnedCharacter {
  root: TransformNode
  animationGroups: AnimationGroup[]
}

interface AnimatedStaticMesh {
  mesh: BabylonMesh
  material: StandardMaterial
  rotationSpeed: [number, number, number]
}

export class BabylonAdapter implements EngineAdapter {
  private engine!: Engine | WebGPUEngine
  private scene!: Scene
  private dirLight!: DirectionalLight
  private shadowGen: ShadowGenerator | null = null
  private boxes: AnimatedStaticMesh[] = []
  private tetrahedra: AnimatedStaticMesh[] = []
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

    if (useCase === 'skinned-mesh') {
      const camera = new ArcRotateCamera('camera', -Math.PI / 4, Math.PI / 3, 50, new Vector3(0, 5, 0), this.scene)
      camera.minZ = 0.1
      camera.attachControl(canvas, true)
    } else {
      const camera = new ArcRotateCamera('camera', -Math.PI / 4, Math.PI / 3, 80, Vector3.Zero(), this.scene)
      camera.minZ = 0.1
      camera.attachControl(canvas, true)
    }

    const ambient = new HemisphericLight('ambient', new Vector3(0, 1, 0), this.scene)
    ambient.intensity = 0.4

    this.dirLight = new DirectionalLight('dirLight', new Vector3(-1, -2, -1), this.scene)
    this.dirLight.position = new Vector3(30, 50, 30)
    this.dirLight.intensity = 1.5

    if (useCase === 'skinned-mesh') {
      this.baseContainer = await SceneLoader.LoadAssetContainerAsync(
        '/models/',
        'Michelle.glb',
        this.scene,
      )
    }

    window.addEventListener('resize', this.onResize)
  }

  private createCustomMesh(name: string, data: BufferGeometryData) {
    const mesh = new BabylonMesh(name, this.scene)
    const vertexData = new VertexData()
    vertexData.positions = Array.from(data.positions)
    vertexData.normals = Array.from(data.normals)
    vertexData.indices = Array.from(data.indices)
    vertexData.applyToMesh(mesh, false)
    return mesh
  }

  private disposeAnimatedMesh(entry: AnimatedStaticMesh) {
    if (this.shadowGen) {
      this.shadowGen.removeShadowCaster(entry.mesh)
    }
    entry.material.dispose()
    entry.mesh.dispose()
  }

  private onResize = () => {
    this.engine.resize()
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
      const box = this.boxes.pop()!
      this.disposeAnimatedMesh(box)
    }

    const spread = 50
    while (this.boxes.length < count) {
      const mesh = MeshBuilder.CreateBox(`cube_${this.boxes.length}`, { size: 1 }, this.scene)
      const material = new StandardMaterial(`mat_${this.boxes.length}`, this.scene)
      material.diffuseColor = new Color3(Math.random(), Math.random(), Math.random())
      material.specularColor = Color3.Black()
      mesh.material = material
      mesh.position = new Vector3(
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
      )

      if (this.shadowsEnabled && this.shadowGen) {
        this.shadowGen.addShadowCaster(mesh)
        mesh.receiveShadows = true
      }

      this.boxes.push({
        mesh,
        material,
        rotationSpeed: [1.0, 0.7, 0],
      })
    }
  }

  private setUniqueTetrahedronCount(count: number) {
    while (this.tetrahedra.length > count) {
      const tetra = this.tetrahedra.pop()!
      this.disposeAnimatedMesh(tetra)
    }

    while (this.tetrahedra.length < count) {
      const index = this.tetrahedra.length
      const spec = createUniqueTetrahedronSpec(index)
      const mesh = this.createCustomMesh(`tetra_${index}`, spec.geometry)
      const material = new StandardMaterial(`tetra_mat_${index}`, this.scene)
      material.diffuseColor = new Color3(...spec.material.color)
      material.specularColor = Color3.Black()
      mesh.material = material
      mesh.position = new Vector3(...spec.position)
      mesh.rotation = new Vector3(...spec.rotation)

      if (this.shadowsEnabled && this.shadowGen) {
        this.shadowGen.addShadowCaster(mesh)
        mesh.receiveShadows = true
      }

      this.tetrahedra.push({
        mesh,
        material,
        rotationSpeed: spec.rotationSpeed,
      })
    }
  }

  private setCharacterCount(count: number) {
    if (!this.baseContainer) return

    while (this.characters.length > count) {
      const char = this.characters.pop()!
      for (const ag of char.animationGroups) {
        ag.stop()
        ag.dispose()
      }
      char.root.dispose()
    }

    const cols = Math.ceil(Math.sqrt(count))
    const spacing = 2
    const containerAnimGroups = this.baseContainer.animationGroups as AnimationGroup[]
    const totalAnimCount = containerAnimGroups.length

    while (this.characters.length < count) {
      const i = this.characters.length
      const instance = this.baseContainer.instantiateModelsToScene(
        (name: string) => `${name}_${i}`,
        false,
      )

      const root = instance.rootNodes[0] as TransformNode
      const animGroups = instance.animationGroups as AnimationGroup[]

      const row = Math.floor(i / cols)
      const col = i % cols
      root.position = new Vector3(
        (col - cols / 2) * spacing,
        0,
        (row - cols / 2) * spacing,
      )
      root.scaling = new Vector3(0.01, 0.01, 0.01)

      const animIndex = totalAnimCount > 1 ? i % totalAnimCount : 0
      for (const ag of animGroups) {
        ag.stop()
      }
      if (animGroups.length > animIndex) {
        const ag = animGroups[animIndex]
        ag.start(true)
        const duration = ag.to - ag.from
        const timeOffset = (i / count) * duration
        ag.goToFrame(ag.from + timeOffset)
      }

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
      for (const box of this.boxes) {
        this.shadowGen.addShadowCaster(box.mesh)
        box.mesh.receiveShadows = true
      }
      for (const tetra of this.tetrahedra) {
        this.shadowGen.addShadowCaster(tetra.mesh)
        tetra.mesh.receiveShadows = true
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
      for (const box of this.boxes) {
        box.mesh.receiveShadows = false
      }
      for (const tetra of this.tetrahedra) {
        tetra.mesh.receiveShadows = false
      }
      for (const char of this.characters) {
        for (const mesh of char.root.getChildMeshes()) {
          mesh.receiveShadows = false
        }
      }
    }
  }

  render(dt: number) {
    const staticMeshes = this.useCase === 'boxes'
      ? this.boxes
      : this.useCase === 'unique-tetrahedra'
        ? this.tetrahedra
        : null

    if (staticMeshes) {
      for (const entry of staticMeshes) {
        entry.mesh.rotation.x += entry.rotationSpeed[0] * dt
        entry.mesh.rotation.y += entry.rotationSpeed[1] * dt
        entry.mesh.rotation.z += entry.rotationSpeed[2] * dt
      }
    }

    this.engine.beginFrame()
    this.scene.render()
    this.engine.endFrame()
  }

  dispose() {
    window.removeEventListener('resize', this.onResize)
    if (this.shadowGen) {
      this.shadowGen.dispose()
      this.shadowGen = null
    }

    for (const box of this.boxes) {
      this.disposeAnimatedMesh(box)
    }
    for (const tetra of this.tetrahedra) {
      this.disposeAnimatedMesh(tetra)
    }
    for (const char of this.characters) {
      for (const ag of char.animationGroups) {
        ag.stop()
        ag.dispose()
      }
      char.root.dispose()
    }

    this.boxes = []
    this.tetrahedra = []
    this.characters = []
    if (this.baseContainer) {
      this.baseContainer.dispose()
      this.baseContainer = null
    }
    this.scene.dispose()
    this.engine.dispose()
  }

  getInfo(): string {
    const type = this.engine instanceof WebGPUEngine ? 'WebGPU' : 'WebGL'
    return `Babylon.js ${type}`
  }
}
