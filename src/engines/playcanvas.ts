import type { EngineAdapter, BackendType, UseCase } from '../types'
import * as pc from 'playcanvas'

interface SkinnedCharacter {
  entity: pc.Entity
  stateGraphAsset?: pc.Asset
}

export class PlayCanvasAdapter implements EngineAdapter {
  private app!: pc.AppBase
  private cubes: pc.Entity[] = []
  private characters: SkinnedCharacter[] = []
  private cameraEntity!: pc.Entity
  private dirLightEntity!: pc.Entity
  private shadowsEnabled = false
  private root!: pc.Entity
  private orbitYaw = -45
  private orbitPitch = 60
  private orbitDistance = 80
  private orbitTarget = new pc.Vec3(0, 0, 0)
  private pointerDown = false
  private lastPointer = { x: 0, y: 0 }
  private useCase: UseCase = 'boxes'
  private containerAsset: pc.Asset | null = null
  private containerResource: any = null
  private animClipNames: string[] = []
  private animClipDurations: number[] = []

  async init(canvas: HTMLCanvasElement, backend: BackendType, useCase: UseCase) {
    this.useCase = useCase

    const deviceTypes = backend === 'webgpu'
      ? [pc.DEVICETYPE_WEBGPU, pc.DEVICETYPE_WEBGL2]
      : [pc.DEVICETYPE_WEBGL2]

    const device = await pc.createGraphicsDevice(canvas, {
      deviceTypes,
      antialias: true,
      powerPreference: 'high-performance',
    })
    device.maxPixelRatio = Math.min(window.devicePixelRatio, 2)

    const createOptions = new pc.AppOptions()
    createOptions.graphicsDevice = device

    const componentSystems: any[] = [
      pc.RenderComponentSystem,
      pc.CameraComponentSystem,
      pc.LightComponentSystem,
    ]

    if (useCase === 'skinned-mesh') {
      componentSystems.push(pc.AnimComponentSystem)
    }

    createOptions.componentSystems = componentSystems
    createOptions.resourceHandlers = [
      pc.TextureHandler,
      pc.ContainerHandler,
      pc.AnimClipHandler,
      pc.AnimStateGraphHandler,
    ]

    const app = new pc.AppBase(canvas)
    app.init(createOptions)
    this.app = app

    app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW)
    app.setCanvasResolution(pc.RESOLUTION_AUTO)

    // Camera
    this.cameraEntity = new pc.Entity('camera')
    this.cameraEntity.addComponent('camera', {
      clearColor: new pc.Color(0, 0, 0, 1),
      farClip: 1000,
      nearClip: 0.1,
      fov: 60,
    })
    app.root.addChild(this.cameraEntity)

    if (useCase === 'boxes') {
      this.orbitDistance = 80
      this.orbitPitch = 60
      this.orbitYaw = -45
      this.orbitTarget.set(0, 0, 0)
    } else {
      this.orbitDistance = 50
      this.orbitPitch = 60
      this.orbitYaw = -45
      this.orbitTarget.set(0, 5, 0)
    }
    this.updateOrbitCamera()

    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerup', this.onPointerUp)
    canvas.addEventListener('wheel', this.onWheel)

    // Ambient light via scene ambient
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

    // Container for meshes
    this.root = new pc.Entity('meshRoot')
    app.root.addChild(this.root)

    app.start()

    if (useCase === 'skinned-mesh') {
      await this.loadModel()
    }

    window.addEventListener('resize', this.onResize)
  }

  private async loadModel() {
    return new Promise<void>((resolve, reject) => {
      this.containerAsset = new pc.Asset('michelle', 'container', {
        url: '/models/Michelle.glb',
      })
      this.containerAsset.on('load', () => {
        this.containerResource = this.containerAsset!.resource

        // Extract animation info
        const animations = this.containerResource.animations
        if (animations) {
          for (let i = 0; i < animations.length; i++) {
            const anim = animations[i]
            this.animClipNames.push(anim.name || `clip_${i}`)
            this.animClipDurations.push(anim.resource?.duration || anim.duration || 1)
          }
        }
        resolve()
      })
      this.containerAsset.on('error', (err: string) => {
        reject(new Error(`Failed to load model: ${err}`))
      })
      this.app.assets.add(this.containerAsset)
      this.app.assets.load(this.containerAsset)
    })
  }

  private updateOrbitCamera() {
    const pitchRad = this.orbitPitch * Math.PI / 180
    const yawRad = this.orbitYaw * Math.PI / 180
    const x = this.orbitTarget.x + this.orbitDistance * Math.cos(pitchRad) * Math.sin(yawRad)
    const y = this.orbitTarget.y + this.orbitDistance * Math.sin(pitchRad)
    const z = this.orbitTarget.z + this.orbitDistance * Math.cos(pitchRad) * Math.cos(yawRad)
    this.cameraEntity.setPosition(x, y, z)
    this.cameraEntity.lookAt(this.orbitTarget)
  }

  private onPointerDown = (e: PointerEvent) => {
    this.pointerDown = true
    this.lastPointer.x = e.clientX
    this.lastPointer.y = e.clientY
  }

  private onPointerMove = (e: PointerEvent) => {
    if (!this.pointerDown) return
    const dx = e.clientX - this.lastPointer.x
    const dy = e.clientY - this.lastPointer.y
    this.lastPointer.x = e.clientX
    this.lastPointer.y = e.clientY
    this.orbitYaw -= dx * 0.3
    this.orbitPitch = Math.max(-89, Math.min(89, this.orbitPitch + dy * 0.3))
    this.updateOrbitCamera()
  }

  private onPointerUp = () => {
    this.pointerDown = false
  }

  private onWheel = (e: WheelEvent) => {
    this.orbitDistance = Math.max(2, this.orbitDistance + e.deltaY * 0.05)
    this.updateOrbitCamera()
  }

  private onResize = () => {
    this.app.resizeCanvas()
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
      cube.destroy()
    }

    const spread = 50
    while (this.cubes.length < count) {
      const entity = new pc.Entity(`cube_${this.cubes.length}`)

      const material = new pc.StandardMaterial()
      material.diffuse = new pc.Color(Math.random(), Math.random(), Math.random())
      material.specular = new pc.Color(0, 0, 0)
      material.gloss = 0
      material.update()

      entity.addComponent('render', {
        type: 'box',
        material,
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

  private setCharacterCount(count: number) {
    if (!this.containerResource) return

    // Remove excess characters
    while (this.characters.length > count) {
      const char = this.characters.pop()!
      char.entity.destroy()
    }

    // Add missing characters
    const cols = Math.ceil(Math.sqrt(count))
    const spacing = 2
    const numAnims = this.animClipNames.length

    while (this.characters.length < count) {
      const i = this.characters.length
      const entity = this.containerResource.instantiateRenderEntity()
      entity.name = `character_${i}`

      // Grid layout
      const row = Math.floor(i / cols)
      const col = i % cols
      entity.setPosition(
        (col - cols / 2) * spacing,
        0,
        (row - cols / 2) * spacing,
      )
      entity.setLocalScale(0.01, 0.01, 0.01)

      // Set shadow properties
      this.setEntityShadows(entity, this.shadowsEnabled)

      this.root.addChild(entity)

      // Set up animation
      if (numAnims > 0) {
        const animIndex = numAnims > 1 ? i % numAnims : 0
        entity.addComponent('anim', {
          activate: true,
          speed: 1,
        })

        // Create a simple state graph with a single dancing state
        const stateGraphData = {
          layers: [{
            name: 'Base',
            states: [
              { name: 'START' },
              { name: 'Dance', speed: 1 },
            ],
            transitions: [{
              from: 'START',
              to: 'Dance',
              time: 0,
              priority: 0,
            }],
          }],
          parameters: {},
        }

        entity.anim!.loadStateGraph(stateGraphData)

        // Assign the animation track to the Dance state
        const animations = this.containerResource.animations
        if (animations && animations[animIndex]) {
          const animTrack = animations[animIndex].resource
          entity.anim!.assignAnimation('Base.Dance', animTrack)

          // Offset the animation time so characters are not in sync
          const duration = this.animClipDurations[animIndex] || 1
          const timeOffset = (i / count) * duration
          const baseLayer = entity.anim!.baseLayer
          if (baseLayer) {
            baseLayer.activeStateCurrentTime = timeOffset
          }
        }
      }

      this.characters.push({ entity })
    }
  }

  private setEntityShadows(entity: pc.Entity, enabled: boolean) {
    const renders = entity.findComponents('render') as pc.RenderComponent[]
    for (const render of renders) {
      render.castShadows = enabled
      render.receiveShadows = enabled
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
    for (const char of this.characters) {
      this.setEntityShadows(char.entity, enabled)
    }
  }

  render(dt: number) {
    if (this.useCase === 'boxes') {
      const speed = 1.0
      for (const cube of this.cubes) {
        cube.rotateLocal(speed * dt * 57.3, speed * dt * 0.7 * 57.3, 0)
      }
    }
    // PlayCanvas handles animation updates internally via the AnimComponent system
  }

  dispose() {
    const canvas = this.app.graphicsDevice.canvas
    canvas.removeEventListener('pointerdown', this.onPointerDown)
    canvas.removeEventListener('pointermove', this.onPointerMove)
    canvas.removeEventListener('pointerup', this.onPointerUp)
    canvas.removeEventListener('wheel', this.onWheel)
    window.removeEventListener('resize', this.onResize)
    for (const cube of this.cubes) {
      cube.destroy()
    }
    for (const char of this.characters) {
      char.entity.destroy()
    }
    this.cubes = []
    this.characters = []
    this.containerAsset = null
    this.containerResource = null
    this.animClipNames = []
    this.animClipDurations = []
    this.app.destroy()
  }

  getInfo(): string {
    const type = this.app.graphicsDevice.deviceType
    return `PlayCanvas (${type})`
  }
}
