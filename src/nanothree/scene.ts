// Scene class for nanothree

import type { Object3D } from './core'
import type { AmbientLight, DirectionalLight } from './light'
import type { Mesh } from './mesh'

export class Scene {
  readonly children: Object3D[] = []
  readonly meshes: Mesh[] = []
  readonly ambientLights: AmbientLight[] = []
  readonly directionalLights: DirectionalLight[] = []

  add(object: Object3D) {
    this.children.push(object)
    // Classify for fast access during rendering
    if ((object as any).geometry && (object as any).material) {
      this.meshes.push(object as Mesh)
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

    if ((object as any).geometry) {
      idx = this.meshes.indexOf(object as Mesh)
      if (idx !== -1) this.meshes.splice(idx, 1)
    }
  }
}
