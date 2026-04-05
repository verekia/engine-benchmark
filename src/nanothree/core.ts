// Core scene graph types for nanothree

import { mat4Perspective, mat4LookAt, mat4Multiply } from './math'

export class Color {
  r: number
  g: number
  b: number

  constructor(rOrHex?: number | Color, g?: number, b?: number) {
    if (rOrHex instanceof Color) {
      this.r = rOrHex.r; this.g = rOrHex.g; this.b = rOrHex.b
    } else if (g !== undefined && b !== undefined) {
      this.r = rOrHex!; this.g = g; this.b = b
    } else if (typeof rOrHex === 'number') {
      this.r = ((rOrHex >> 16) & 0xff) / 255
      this.g = ((rOrHex >> 8) & 0xff) / 255
      this.b = (rOrHex & 0xff) / 255
    } else {
      this.r = 1; this.g = 1; this.b = 1
    }
  }

  set(rOrHex: number | Color, g?: number, b?: number) {
    if (rOrHex instanceof Color) {
      this.r = rOrHex.r; this.g = rOrHex.g; this.b = rOrHex.b
    } else if (g !== undefined && b !== undefined) {
      this.r = rOrHex; this.g = g; this.b = b
    } else {
      this.r = ((rOrHex >> 16) & 0xff) / 255
      this.g = ((rOrHex >> 8) & 0xff) / 255
      this.b = (rOrHex & 0xff) / 255
    }
    return this
  }
}

export class Vector3 {
  x: number
  y: number
  z: number

  constructor(x = 0, y = 0, z = 0) {
    this.x = x; this.y = y; this.z = z
  }

  set(x: number, y: number, z: number) {
    this.x = x; this.y = y; this.z = z
    return this
  }
}

export class Euler {
  x: number
  y: number
  z: number

  constructor(x = 0, y = 0, z = 0) {
    this.x = x; this.y = y; this.z = z
  }

  set(x: number, y: number, z: number) {
    this.x = x; this.y = y; this.z = z
    return this
  }
}

export class Object3D {
  readonly position = new Vector3()
  readonly rotation = new Euler()
  readonly scale = new Vector3(1, 1, 1)
  visible = true
  castShadow = false
  receiveShadow = false
}

export class PerspectiveCamera extends Object3D {
  aspect: number
  private _fov: number
  private _near: number
  private _far: number

  private proj = new Float32Array(16)
  private view = new Float32Array(16)
  readonly viewProjection = new Float32Array(16)
  private target = new Vector3()

  get fov() { return this._fov * (180 / Math.PI) }
  get near() { return this._near }
  get far() { return this._far }

  constructor(fov = 50, aspect = 1, near = 0.1, far = 2000) {
    super()
    this._fov = fov * (Math.PI / 180)
    this.aspect = aspect
    this._near = near
    this._far = far
  }

  updateProjectionMatrix() {
    // Projection will be recalculated on next updateViewProjection
  }

  updateViewProjection(aspect?: number) {
    if (aspect !== undefined) this.aspect = aspect
    mat4Perspective(this.proj, this._fov, this.aspect, this._near, this._far)
    mat4LookAt(
      this.view,
      this.position.x, this.position.y, this.position.z,
      this.target.x, this.target.y, this.target.z,
      0, 1, 0,
    )
    mat4Multiply(this.viewProjection, this.proj, this.view)
  }
}
