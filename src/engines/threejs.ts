import type { EngineAdapter, BackendType, UseCase } from '../types'
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
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js'

interface SkinnedCharacter {
  model: Object3D
  mixer: AnimationMixer
}

export class ThreeAdapter implements EngineAdapter {
  private renderer!: WebGPURenderer
  private scene!: Scene
  private camera!: PerspectiveCamera
  private cubes: Mesh[] = []
  private characters: SkinnedCharacter[] = []
  private dirLight!: DirectionalLight
  private geometry!: BoxGeometry
  private shadowsEnabled = false
  private useCase: UseCase = 'boxes'
  private baseModel: Group | null = null
  private animations: AnimationClip[] = []

  async init(canvas: HTMLCanvasElement, backend: BackendType, useCase: UseCase) {
    this.useCase = useCase
    this.scene = new Scene()
    this.camera = new PerspectiveCamera(60, canvas.width / canvas.height, 0.1, 1000)
    this.camera.position.set(0, 40, 60)
    this.camera.lookAt(0, 0, 0)

    this.renderer = new WebGPURenderer({
      canvas,
      antialias: true,
      forceWebGL: backend === 'webgl',
    })
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false)
    this.renderer.setPixelRatio(window.devicePixelRatio)
    this.renderer.shadowMap.enabled = false

    await this.renderer.init()

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
      this.geometry = new BoxGeometry(1, 1, 1)
    } else {
      // Load the skinned mesh model
      const loader = new GLTFLoader()
      const gltf = await loader.loadAsync('/models/Michelle.glb')
      this.baseModel = gltf.scene
      this.animations = gltf.animations
      // Scale the model to a reasonable size
      this.baseModel.scale.set(0.01, 0.01, 0.01)

      // Set up camera for character viewing
      this.camera.position.set(0, 15, 40)
      this.camera.lookAt(0, 5, 0)
    }

    window.addEventListener('resize', this.onResize)
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
    } else {
      this.setCharacterCount(count)
    }
  }

  private setBoxCount(count: number) {
    while (this.cubes.length > count) {
      const cube = this.cubes.pop()!
      this.scene.remove(cube)
      ;(cube.material as MeshLambertMaterial).dispose()
    }

    const spread = 50
    while (this.cubes.length < count) {
      const color = new Color(Math.random(), Math.random(), Math.random())
      const mat = new MeshLambertMaterial({ color })
      const mesh = new Mesh(this.geometry, mat)
      mesh.position.set(
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
    if (!this.baseModel || this.animations.length === 0) return

    // Remove excess characters
    while (this.characters.length > count) {
      const char = this.characters.pop()!
      char.mixer.stopAllAction()
      this.scene.remove(char.model)
    }

    // Add missing characters
    const cols = Math.ceil(Math.sqrt(count))
    const spacing = 2
    while (this.characters.length < count) {
      const i = this.characters.length
      const clone = SkeletonUtils.clone(this.baseModel)

      // Grid layout
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

      // Alternate animations if multiple exist, otherwise just use the first
      const clipIndex = this.animations.length > 1 ? i % this.animations.length : 0
      const action = mixer.clipAction(this.animations[clipIndex])
      action.play()

      // Offset animation time so characters are not in sync
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

    for (const cube of this.cubes) {
      cube.castShadow = enabled
      cube.receiveShadow = enabled
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
    if (this.useCase === 'boxes') {
      const speed = 1.0
      for (const cube of this.cubes) {
        cube.rotation.x += speed * dt
        cube.rotation.y += speed * dt * 0.7
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
    for (const cube of this.cubes) {
      (cube.material as MeshLambertMaterial).dispose()
    }
    for (const char of this.characters) {
      char.mixer.stopAllAction()
      this.scene.remove(char.model)
    }
    if (this.geometry) this.geometry.dispose()
    this.renderer.dispose()
    this.cubes = []
    this.characters = []
    this.baseModel = null
    this.animations = []
  }

  getInfo(): string {
    return `Three.js r${(this.renderer as any).info?.render ? '' : ''}${this.renderer.constructor.name}`
  }
}
