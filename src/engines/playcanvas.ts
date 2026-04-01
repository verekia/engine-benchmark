import type { EngineAdapter, BackendType } from '../types'
import * as pc from 'playcanvas'

export class PlayCanvasAdapter implements EngineAdapter {
  private app!: pc.Application
  private cubes: pc.Entity[] = []
  private dirLightEntity!: pc.Entity
  private shadowsEnabled = false
  private root!: pc.Entity

  async init(canvas: HTMLCanvasElement, backend: BackendType) {
    const deviceTypes = backend === 'webgpu'
      ? [pc.DEVICETYPE_WEBGPU, pc.DEVICETYPE_WEBGL2]
      : [pc.DEVICETYPE_WEBGL2]

    const device = await pc.createGraphicsDevice(canvas, {
      deviceTypes,
      glslangUrl: undefined as any,
      twgslUrl: undefined as any,
    })

    const app = new pc.Application(canvas, {
      graphicsDevice: device,
    })
    this.app = app

    app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW)
    app.setCanvasResolution(pc.RESOLUTION_AUTO)

    // Camera
    const camera = new pc.Entity('camera')
    camera.addComponent('camera', {
      clearColor: new pc.Color(0, 0, 0, 1),
      farClip: 1000,
      nearClip: 0.1,
      fov: 60,
    })
    camera.setPosition(0, 40, 60)
    camera.lookAt(0, 0, 0)
    app.root.addChild(camera)

    // Ambient light (hemispheric via scene ambient)
    app.scene.ambientLight = new pc.Color(0.25, 0.25, 0.25)

    // Directional light
    this.dirLightEntity = new pc.Entity('dirLight')
    this.dirLightEntity.addComponent('light', {
      type: 'directional',
      color: new pc.Color(1, 1, 1),
      intensity: 1.5,
      castShadows: false,
      shadowResolution: 2048,
      shadowDistance: 120,
      shadowBias: 0.2,
      normalOffsetBias: 0.05,
    })
    this.dirLightEntity.setEulerAngles(45, 30, 0)
    app.root.addChild(this.dirLightEntity)

    // Container for cubes
    this.root = new pc.Entity('cubeRoot')
    app.root.addChild(this.root)

    app.start()

    window.addEventListener('resize', this.onResize)
  }

  private onResize = () => {
    this.app.resizeCanvas()
  }

  setCubeCount(count: number) {
    while (this.cubes.length > count) {
      const cube = this.cubes.pop()!
      cube.destroy()
    }

    const spread = 50
    while (this.cubes.length < count) {
      const entity = new pc.Entity(`cube_${this.cubes.length}`)

      const material = new pc.StandardMaterial()
      material.diffuse = new pc.Color(Math.random(), Math.random(), Math.random())
      material.update()

      // Create a box mesh
      const mesh = pc.Mesh.fromGeometry(this.app.graphicsDevice, new pc.BoxGeometry())
      const meshInstance = new pc.MeshInstance(mesh, material)
      meshInstance.castShadow = this.shadowsEnabled

      entity.addComponent('render', {
        meshInstances: [meshInstance],
        castShadows: this.shadowsEnabled,
        receiveShadows: this.shadowsEnabled,
      })

      entity.setPosition(
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
      )

      this.root.addChild(entity)
      this.cubes.push(entity)
    }
  }

  setShadows(enabled: boolean) {
    this.shadowsEnabled = enabled
    this.dirLightEntity.light!.castShadows = enabled
    for (const cube of this.cubes) {
      if (cube.render) {
        cube.render.castShadows = enabled
        cube.render.receiveShadows = enabled
      }
    }
  }

  render(dt: number) {
    const speed = 1.0
    for (const cube of this.cubes) {
      cube.rotateLocal(speed * dt * 57.3, speed * dt * 0.7 * 57.3, 0)
    }
    // PlayCanvas has its own internal render loop via app.start()
  }

  dispose() {
    window.removeEventListener('resize', this.onResize)
    for (const cube of this.cubes) {
      cube.destroy()
    }
    this.cubes = []
    this.app.destroy()
  }

  getInfo(): string {
    const type = this.app.graphicsDevice.deviceType
    return `PlayCanvas (${type})`
  }
}
