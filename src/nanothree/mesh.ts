// Mesh class for nanothree

import { Object3D } from './core'
import type { BufferGeometry } from './geometry'
import type { MeshLambertMaterial } from './material'

export class Mesh extends Object3D {
  constructor(
    public geometry: BufferGeometry,
    public material: MeshLambertMaterial,
  ) {
    super()
  }
}
