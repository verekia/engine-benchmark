// Scene class for nanothree

import type { Object3D } from './core'
import type { AmbientLight, DirectionalLight } from './light'
import type { Mesh } from './mesh'
import type { Line } from './line'

export class Scene {
  readonly children: Object3D[] = []
  readonly meshes: Mesh[] = []
  readonly lines: Line[] = []
  readonly ambientLights: AmbientLight[] = []
  readonly directionalLights: DirectionalLight[] = []

  add(object: Object3D) {
    this.children.push(object)
    if ((object as any).isMesh) {
      this.meshes.push(object as Mesh)
    } else if ((object as any).isLine) {
      this.lines.push(object as Line)
    } else if ((object as any).intensity !== undefined) {
      if ((object as any).shadow) {
        this.directionalLights.push(object as DirectionalLight)
      } else {
        this.ambientLights.push(object as AmbientLight)
      }
    }
  }

  remove(object: Object3D) {
    let idx = this.children.indexOf(object)
    if (idx !== -1) this.children.splice(idx, 1)

    if ((object as any).isMesh) {
      idx = this.meshes.indexOf(object as Mesh)
      if (idx !== -1) this.meshes.splice(idx, 1)
    } else if ((object as any).isLine) {
      idx = this.lines.indexOf(object as Line)
      if (idx !== -1) this.lines.splice(idx, 1)
    }
  }
}
