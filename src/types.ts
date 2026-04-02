export type EngineName = 'threejs' | 'playcanvas' | 'babylonjs' | 'voidcore' | 'experiment-a' | 'experiment-b' | 'experiment-c'
export type BackendType = 'webgl' | 'webgpu'
export type UseCase = 'boxes' | 'skinned-mesh'

export interface BenchmarkParams {
  engine: EngineName
  backend: BackendType
  useCase: UseCase
  meshCount: number
  shadows: boolean
}

export interface EngineAdapter {
  init(canvas: HTMLCanvasElement, backend: BackendType, useCase: UseCase): Promise<void>
  setMeshCount(count: number): void
  setShadows(enabled: boolean): void
  render(dt: number): void
  dispose(): void
  getInfo(): string
}
