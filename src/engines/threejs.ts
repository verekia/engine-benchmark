import type { EngineAdapter, BackendType } from '../types'
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
} from 'three/webgpu'

export class ThreeAdapter implements EngineAdapter {
  private renderer!: WebGPURenderer
  private scene!: Scene
  private camera!: PerspectiveCamera
  private cubes: Mesh[] = []
  private dirLight!: DirectionalLight
  private geometry!: BoxGeometry
  private shadowsEnabled = false

  async init(canvas: HTMLCanvasElement, backend: BackendType) {
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

    this.geometry = new BoxGeometry(1, 1, 1)

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

  setCubeCount(count: number) {
    // Remove excess cubes
    while (this.cubes.length > count) {
      const cube = this.cubes.pop()!
      this.scene.remove(cube)
      ;(cube.material as MeshLambertMaterial).dispose()
    }

    // Add missing cubes
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

  setShadows(enabled: boolean) {
    this.shadowsEnabled = enabled
    this.renderer.shadowMap.enabled = enabled
    this.dirLight.castShadow = enabled
    for (const cube of this.cubes) {
      cube.castShadow = enabled
      cube.receiveShadow = enabled
    }
  }

  render(dt: number) {
    const speed = 1.0
    for (const cube of this.cubes) {
      cube.rotation.x += speed * dt
      cube.rotation.y += speed * dt * 0.7
    }
    this.renderer.render(this.scene, this.camera)
  }

  dispose() {
    window.removeEventListener('resize', this.onResize)
    for (const cube of this.cubes) {
      (cube.material as MeshLambertMaterial).dispose()
    }
    this.geometry.dispose()
    this.renderer.dispose()
    this.cubes = []
  }

  getInfo(): string {
    return `Three.js r${(this.renderer as any).info?.render ? '' : ''}${this.renderer.constructor.name}`
  }
}
