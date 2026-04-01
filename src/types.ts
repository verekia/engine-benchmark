export type EngineName = 'threejs' | 'playcanvas' | 'babylonjs' | 'voidcore'
export type BackendType = 'webgl' | 'webgpu'

export interface BenchmarkParams {
  engine: EngineName
  backend: BackendType
  cubeCount: number
  shadows: boolean
}

export interface EngineAdapter {
  init(canvas: HTMLCanvasElement, backend: BackendType): Promise<void>
  setCubeCount(count: number): void
  setShadows(enabled: boolean): void
  render(dt: number): void
  dispose(): void
  getInfo(): string
}
