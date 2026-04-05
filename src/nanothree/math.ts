// Lightweight math utilities for nanothree

export function mat4Perspective(
  out: Float32Array,
  fov: number,
  aspect: number,
  near: number,
  far: number,
) {
  const f = 1 / Math.tan(fov / 2)
  const nf = 1 / (near - far)
  out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0
  out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0
  out[8] = 0; out[9] = 0; out[10] = far * nf; out[11] = -1
  out[12] = 0; out[13] = 0; out[14] = near * far * nf; out[15] = 0
}

export function mat4LookAt(
  out: Float32Array,
  ex: number, ey: number, ez: number,
  tx: number, ty: number, tz: number,
  ux: number, uy: number, uz: number,
) {
  let fx = tx - ex, fy = ty - ey, fz = tz - ez
  let len = 1 / Math.sqrt(fx * fx + fy * fy + fz * fz)
  fx *= len; fy *= len; fz *= len
  let rx = fy * uz - fz * uy, ry = fz * ux - fx * uz, rz = fx * uy - fy * ux
  len = 1 / Math.sqrt(rx * rx + ry * ry + rz * rz)
  rx *= len; ry *= len; rz *= len
  const sx = ry * fz - rz * fy, sy = rz * fx - rx * fz, sz = rx * fy - ry * fx
  out[0] = rx; out[1] = sx; out[2] = -fx; out[3] = 0
  out[4] = ry; out[5] = sy; out[6] = -fy; out[7] = 0
  out[8] = rz; out[9] = sz; out[10] = -fz; out[11] = 0
  out[12] = -(rx * ex + ry * ey + rz * ez)
  out[13] = -(sx * ex + sy * ey + sz * ez)
  out[14] = fx * ex + fy * ey + fz * ez
  out[15] = 1
}

export function mat4Multiply(out: Float32Array, a: Float32Array, b: Float32Array) {
  for (let i = 0; i < 4; i++) {
    const ai0 = a[i], ai1 = a[i + 4], ai2 = a[i + 8], ai3 = a[i + 12]
    out[i]      = ai0 * b[0]  + ai1 * b[1]  + ai2 * b[2]  + ai3 * b[3]
    out[i + 4]  = ai0 * b[4]  + ai1 * b[5]  + ai2 * b[6]  + ai3 * b[7]
    out[i + 8]  = ai0 * b[8]  + ai1 * b[9]  + ai2 * b[10] + ai3 * b[11]
    out[i + 12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15]
  }
}

// WebGPU orthographic projection (Z maps to [0, 1])
export function mat4Ortho(
  out: Float32Array,
  left: number, right: number,
  bottom: number, top: number,
  near: number, far: number,
) {
  const lr = 1 / (left - right)
  const bt = 1 / (bottom - top)
  const nf = 1 / (near - far)
  out[0] = -2 * lr; out[1] = 0; out[2] = 0; out[3] = 0
  out[4] = 0; out[5] = -2 * bt; out[6] = 0; out[7] = 0
  out[8] = 0; out[9] = 0; out[10] = nf; out[11] = 0
  out[12] = (left + right) * lr
  out[13] = (top + bottom) * bt
  out[14] = near * nf
  out[15] = 1
}

export function mat4FromEulerXYZ(out: Float32Array, rx: number, ry: number, rz: number) {
  const cx = Math.cos(rx), sx = Math.sin(rx)
  const cy = Math.cos(ry), sy = Math.sin(ry)
  const cz = Math.cos(rz), sz = Math.sin(rz)
  out[0] = cz * cy
  out[1] = sz * cy
  out[2] = -sy
  out[3] = 0
  out[4] = cz * sy * sx - sz * cx
  out[5] = sz * sy * sx + cz * cx
  out[6] = cy * sx
  out[7] = 0
  out[8] = cz * sy * cx + sz * sx
  out[9] = sz * sy * cx - cz * sx
  out[10] = cy * cx
  out[11] = 0
  out[12] = 0
  out[13] = 0
  out[14] = 0
  out[15] = 1
}
