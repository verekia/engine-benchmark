// Material classes for nanothree

import { Color } from './core'

export class MeshLambertMaterial {
  color: Color
  wireframe: boolean

  constructor(params?: { color?: Color | number; wireframe?: boolean }) {
    if (params?.color instanceof Color) {
      this.color = params.color
    } else if (typeof params?.color === 'number') {
      this.color = new Color(params.color)
    } else {
      this.color = new Color(0xffffff)
    }
    this.wireframe = params?.wireframe ?? false
  }

  dispose() {
    // No GPU resources to free for materials
  }
}

export class LineBasicMaterial {
  color: Color

  constructor(params?: { color?: Color | number }) {
    if (params?.color instanceof Color) {
      this.color = params.color
    } else if (typeof params?.color === 'number') {
      this.color = new Color(params.color)
    } else {
      this.color = new Color(0xffffff)
    }
  }

  dispose() {}
}
