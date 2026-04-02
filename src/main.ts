import GUI from 'lil-gui'
import type { EngineName, BackendType, UseCase, EngineAdapter } from './types'

let canvas = document.getElementById('canvas') as HTMLCanvasElement
const statsEl = document.getElementById('stats') as HTMLDivElement

let currentAdapter: EngineAdapter | null = null
let animFrameId = 0
let lastTime = 0

const params = {
  engine: 'threejs' as EngineName,
  backend: 'webgpu' as BackendType,
  useCase: 'boxes' as UseCase,
  meshCount: 1000,
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
    case 'experiment-a': {
      const { ExperimentAAdapter } = await import('./engines/experiment-a')
      return new ExperimentAAdapter()
    }
    case 'experiment-b': {
      const { ExperimentBAdapter } = await import('./engines/experiment-b')
      return new ExperimentBAdapter()
    }
    case 'experiment-c': {
      const { ExperimentCAdapter } = await import('./engines/experiment-c')
      return new ExperimentCAdapter()
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
  newCanvas.style.cssText = canvas.style.cssText
  parent.replaceChild(newCanvas, canvas)
  canvas = newCanvas

  statsEl.textContent = `Loading ${params.engine} (${params.backend}) – ${params.useCase}...`

  // Wait one frame so the browser lays out the new canvas (clientWidth/Height > 0)
  await new Promise(resolve => requestAnimationFrame(resolve))

  try {
    const adapter = await loadAdapter(params.engine)
    await adapter.init(canvas, params.backend, params.useCase)
    currentAdapter = adapter

    adapter.setMeshCount(params.meshCount)
    adapter.setShadows(params.shadows)

    statsEl.textContent = adapter.getInfo()

    // Start render loop
    lastTime = performance.now()
    let frameCount = 0
    let fpsTime = 0

    const meshLabel = params.useCase === 'boxes'
      ? 'cubes'
      : params.useCase === 'skinned-mesh'
        ? 'characters'
        : 'tetrahedra'

    const loop = (now: number) => {
      animFrameId = requestAnimationFrame(loop)
      const dt = Math.min((now - lastTime) / 1000, 0.1)
      lastTime = now

      adapter.render(dt)

      frameCount++
      fpsTime += dt
      if (fpsTime >= 0.5) {
        const fps = Math.round(frameCount / fpsTime)
        statsEl.textContent = `${adapter.getInfo()} | ${params.meshCount} ${meshLabel} | FPS: ${fps}`
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

gui.add(params, 'engine', ['threejs', 'playcanvas', 'babylonjs', 'voidcore', 'experiment-a', 'experiment-b', 'experiment-c']).name('Engine').onChange(() => restart())
gui.add(params, 'backend', ['webgl', 'webgpu']).name('Backend').onChange(() => restart())
gui.add(params, 'useCase', ['boxes', 'skinned-mesh', 'unique-tetrahedra']).name('Use Case').onChange(() => {
  if (params.useCase === 'skinned-mesh') {
    meshCountCtrl.options([100, 500, 1000, 2000])
    if (params.meshCount > 2000) {
      params.meshCount = 1000
      meshCountCtrl.updateDisplay()
    }
  } else if (params.useCase === 'unique-tetrahedra') {
    meshCountCtrl.options([100, 250, 500, 1000, 2000, 5000, 10000, 15000, 20000])
    if (params.meshCount > 20000 || params.meshCount < 100) {
      params.meshCount = 1000
      meshCountCtrl.updateDisplay()
    }
  } else {
    meshCountCtrl.options([1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000])
    if (params.meshCount < 1000) {
      params.meshCount = 1000
      meshCountCtrl.updateDisplay()
    }
  }
  restart()
})

const meshCountCtrl = gui.add(params, 'meshCount', [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]).name('Mesh Count').onChange((v: number) => {
  if (currentAdapter) currentAdapter.setMeshCount(v)
})
gui.add(params, 'shadows').name('Shadows').onChange((v: boolean) => {
  if (currentAdapter) currentAdapter.setShadows(v)
})

// Start
restart()
