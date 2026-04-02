import type { EngineAdapter, BackendType, UseCase } from '../types'
import { createUniqueTetrahedronSpec, type BufferGeometryData } from '../benchmark-scene'
import {
  Scene,
  PerspectiveCamera,
  BoxGeometry,
  MeshLambertMaterial,
  Mesh,
  AmbientLight,
  DirectionalLight,
  Color,
  WebGPURenderer,
  AnimationMixer,
  AnimationClip,
  Object3D,
  Group,
} from 'three/webgpu'
import { BufferGeometry, Float32BufferAttribute } from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js'

interface SkinnedCharacter {
  model: Object3D
  mixer: AnimationMixer
}

interface AnimatedStaticMesh {
  mesh: Mesh
  material: MeshLambertMaterial
  geometry: BufferGeometry
  rotationSpeed: [number, number, number]
}

export class ThreeAdapter implements EngineAdapter {
  private renderer!: WebGPURenderer
  private scene!: Scene
  private camera!: PerspectiveCamera
  private boxes: AnimatedStaticMesh[] = []
  private tetrahedra: AnimatedStaticMesh[] = []
  private characters: SkinnedCharacter[] = []
  private controls!: OrbitControls
  private dirLight!: DirectionalLight
  private boxGeometry: BoxGeometry | null = null
  private shadowsEnabled = false
  private useCase: UseCase = 'boxes'
  private baseModel: Group | null = null
  private animations: AnimationClip[] = []

  async init(canvas: HTMLCanvasElement, backend: BackendType, useCase: UseCase) {
    this.useCase = useCase
    this.scene = new Scene()
    this.camera = new PerspectiveCamera(60, canvas.width / canvas.height, 0.1, 1000)
    this.camera.position.set(0, 40, 60)

    this.renderer = new WebGPURenderer({
      canvas,
      antialias: true,
      forceWebGL: backend === 'webgl',
    })
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false)
    this.renderer.setPixelRatio(window.devicePixelRatio)
    this.renderer.shadowMap.enabled = false

    await this.renderer.init()

    this.controls = new OrbitControls(this.camera, canvas)

    const ambient = new AmbientLight(0x404040, 1)
    this.scene.add(ambient)

    this.dirLight = new DirectionalLight(0xffffff, 1.5)
    this.dirLight.position.set(30, 50, 30)
    this.dirLight.castShadow = false
    this.dirLight.shadow.mapSize.set(2048, 2048)
    this.dirLight.shadow.camera.near = 0.5
    this.dirLight.shadow.camera.far = 200
    this.dirLight.shadow.camera.left = -60
    this.dirLight.shadow.camera.right = 60
    this.dirLight.shadow.camera.top = 60
    this.dirLight.shadow.camera.bottom = -60
    this.scene.add(this.dirLight)

    if (useCase === 'boxes') {
      this.boxGeometry = new BoxGeometry(1, 1, 1)
    } else if (useCase === 'skinned-mesh') {
      const loader = new GLTFLoader()
      const gltf = await loader.loadAsync('/models/Michelle.glb')
      this.baseModel = gltf.scene
      this.animations = gltf.animations
      this.baseModel.scale.set(0.01, 0.01, 0.01)

      this.camera.position.set(0, 15, 40)
      this.controls.target.set(0, 5, 0)
    }

    window.addEventListener('resize', this.onResize)
  }

  private createBufferGeometry(data: BufferGeometryData) {
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new Float32BufferAttribute(data.positions, 3))
    geometry.setAttribute('normal', new Float32BufferAttribute(data.normals, 3))
    geometry.setIndex(Array.from(data.indices))
    return geometry
  }

  private disposeAnimatedMesh(entry: AnimatedStaticMesh, disposeGeometry: boolean) {
    this.scene.remove(entry.mesh)
    entry.material.dispose()
    if (disposeGeometry) {
      entry.geometry.dispose()
    }
  }

  private onResize = () => {
    const canvas = this.renderer.domElement
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h, false)
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
      this.disposeAnimatedMesh(box, false)
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
      this.disposeAnimatedMesh(tetra, true)
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

  private setCharacterCount(count: number) {
    if (!this.baseModel || this.animations.length === 0) return

    while (this.characters.length > count) {
      const char = this.characters.pop()!
      char.mixer.stopAllAction()
      this.scene.remove(char.model)
    }

    const cols = Math.ceil(Math.sqrt(count))
    const spacing = 2
    while (this.characters.length < count) {
      const i = this.characters.length
      const clone = SkeletonUtils.clone(this.baseModel)

      const row = Math.floor(i / cols)
      const col = i % cols
      clone.position.set(
        (col - cols / 2) * spacing,
        0,
        (row - cols / 2) * spacing,
      )

      clone.castShadow = this.shadowsEnabled
      clone.receiveShadow = this.shadowsEnabled
      clone.traverse((child) => {
        if ((child as Mesh).isMesh) {
          child.castShadow = this.shadowsEnabled
          child.receiveShadow = this.shadowsEnabled
        }
      })

      this.scene.add(clone)

      const mixer = new AnimationMixer(clone)
      const clipIndex = this.animations.length > 1 ? i % this.animations.length : 0
      const action = mixer.clipAction(this.animations[clipIndex])
      action.play()

      const duration = this.animations[clipIndex].duration
      const timeOffset = (i / count) * duration
      mixer.setTime(timeOffset)

      this.characters.push({ model: clone, mixer })
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
    for (const char of this.characters) {
      char.model.traverse((child) => {
        if ((child as Mesh).isMesh) {
          child.castShadow = enabled
          child.receiveShadow = enabled
        }
      })
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
    } else {
      for (const char of this.characters) {
        char.mixer.update(dt)
      }
    }

    this.renderer.render(this.scene, this.camera)
  }

  dispose() {
    window.removeEventListener('resize', this.onResize)
    this.controls.dispose()

    for (const box of this.boxes) {
      this.disposeAnimatedMesh(box, false)
    }
    for (const tetra of this.tetrahedra) {
      this.disposeAnimatedMesh(tetra, true)
    }
    for (const char of this.characters) {
      char.mixer.stopAllAction()
      this.scene.remove(char.model)
    }

    this.boxes = []
    this.tetrahedra = []
    this.characters = []
    this.baseModel = null
    this.animations = []
    this.boxGeometry?.dispose()
    this.boxGeometry = null
    this.renderer.dispose()
  }

  getInfo(): string {
    return `Three.js r${(this.renderer as any).info?.render ? '' : ''}${this.renderer.constructor.name}`
  }
}
