// Light classes for nanothree

import { Object3D, Color } from './core'

export class AmbientLight extends Object3D {
  color: Color
  intensity: number

  constructor(color: number | Color = 0xffffff, intensity = 1) {
    super()
    this.color = color instanceof Color ? color : new Color(color)
    this.intensity = intensity
  }
}

export class DirectionalLight extends Object3D {
  color: Color
  intensity: number
  readonly shadow = {
    mapSize: { set(_w: number, _h: number) {} },
    camera: { near: 0.5, far: 200, left: -60, right: 60, top: 60, bottom: -60 },
  }

  constructor(color: number | Color = 0xffffff, intensity = 1) {
    super()
    this.color = color instanceof Color ? color : new Color(color)
    this.intensity = intensity
  }
}
