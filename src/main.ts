import GUI from 'lil-gui'
import type { EngineName, BackendType, EngineAdapter } from './types'

const canvas = document.getElementById('canvas') as HTMLCanvasElement
const statsEl = document.getElementById('stats') as HTMLDivElement

let currentAdapter: EngineAdapter | null = null
let animFrameId = 0
let lastTime = 0

const params = {
  engine: 'threejs' as EngineName,
  backend: 'webgpu' as BackendType,
  cubeCount: 500,
  shadows: false,
}

async function loadAdapter(engine: EngineName): Promise<EngineAdapter> {
  switch (engine) {
    case 'threejs': {
      const { ThreeAdapter } = await import('./engines/threejs')
      return new ThreeAdapter()
    }
    case 'babylonjs': {
      const { BabylonAdapter } = await import('./engines/babylonjs')
      return new BabylonAdapter()
    }
    case 'playcanvas': {
      const { PlayCanvasAdapter } = await import('./engines/playcanvas')
      return new PlayCanvasAdapter()
    }
    case 'voidcore': {
      const { VoidcoreAdapter } = await import('./engines/voidcore')
      return new VoidcoreAdapter()
    }
  }
}

async function restart() {
  // Stop current
  if (animFrameId) {
    cancelAnimationFrame(animFrameId)
    animFrameId = 0
  }
  if (currentAdapter) {
    currentAdapter.dispose()
    currentAdapter = null
  }

  // Reset canvas (clear any WebGL/WebGPU context by replacing)
  const parent = canvas.parentElement!
  const newCanvas = document.createElement('canvas')
  newCanvas.id = 'canvas'
  parent.replaceChild(newCanvas, canvas)
  // Update reference
  ;(document.getElementById('canvas') as HTMLCanvasElement).style.cssText = canvas.style.cssText
  const activeCanvas = document.getElementById('canvas') as HTMLCanvasElement

  statsEl.textContent = `Loading ${params.engine} (${params.backend})...`

  try {
    const adapter = await loadAdapter(params.engine)
    await adapter.init(activeCanvas, params.backend)
    currentAdapter = adapter

    adapter.setCubeCount(params.cubeCount)
    adapter.setShadows(params.shadows)

    statsEl.textContent = adapter.getInfo()

    // Start render loop
    lastTime = performance.now()
    let frameCount = 0
    let fpsTime = 0

    const loop = (now: number) => {
      animFrameId = requestAnimationFrame(loop)
      const dt = Math.min((now - lastTime) / 1000, 0.1)
      lastTime = now

      adapter.render(dt)

      frameCount++
      fpsTime += dt
      if (fpsTime >= 0.5) {
        const fps = Math.round(frameCount / fpsTime)
        statsEl.textContent = `${adapter.getInfo()} | ${params.cubeCount} cubes | FPS: ${fps}`
        frameCount = 0
        fpsTime = 0
      }
    }
    animFrameId = requestAnimationFrame(loop)
  } catch (e) {
    statsEl.textContent = `Error: ${(e as Error).message}`
    console.error(e)
  }
}

// GUI
const gui = new GUI()
gui.domElement.style.position = 'fixed'
gui.domElement.style.top = '0'
gui.domElement.style.right = '0'

gui.add(params, 'engine', ['threejs', 'playcanvas', 'babylonjs', 'voidcore']).name('Engine').onChange(() => restart())
gui.add(params, 'backend', ['webgl', 'webgpu']).name('Backend').onChange(() => restart())
gui.add(params, 'cubeCount', 100, 5000, 100).name('Cube Count').onChange((v: number) => {
  if (currentAdapter) currentAdapter.setCubeCount(v)
})
gui.add(params, 'shadows').name('Shadows').onChange((v: boolean) => {
  if (currentAdapter) currentAdapter.setShadows(v)
})

// Start
restart()
