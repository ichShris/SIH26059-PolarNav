// South polar stereographic (spherical EPSG:3031, true scale 71S) — mirrors backend/polarnav/projection.py
const R = 6371.0
const K0 = (1 + Math.sin((71 * Math.PI) / 180)) / 2
export const TWO_RK = 2 * R * K0
const D2R = Math.PI / 180

export function toXY(lat: number, lon: number): [number, number] {
  const rho = TWO_RK * Math.tan(Math.PI / 4 + (lat * D2R) / 2)
  return [rho * Math.sin(lon * D2R), rho * Math.cos(lon * D2R)]
}

export function toLatLon(x: number, y: number): [number, number] {
  const rho = Math.hypot(x, y)
  const lat = (2 * Math.atan(rho / TWO_RK) - Math.PI / 2) / D2R
  const lon = Math.atan2(x, y) / D2R
  return [lat, lon]
}

export function fmtLat(lat: number) {
  const a = Math.abs(lat)
  const d = Math.floor(a)
  const m = (a - d) * 60
  return `${d}°${m.toFixed(1).padStart(4, '0')}′${lat < 0 ? 'S' : 'N'}`
}

export function fmtLon(lon: number) {
  const l = ((lon + 540) % 360) - 180
  const a = Math.abs(l)
  const d = Math.floor(a)
  const m = (a - d) * 60
  return `${d}°${m.toFixed(1).padStart(4, '0')}′${l < 0 ? 'W' : 'E'}`
}

export interface View { cx: number; cy: number; scale: number; w: number; h: number }

export const toScreen = (v: View, x: number, y: number): [number, number] => [
  v.w / 2 + (x - v.cx) * v.scale,
  v.h / 2 - (y - v.cy) * v.scale,
]

export const toWorld = (v: View, sx: number, sy: number): [number, number] => [
  v.cx + (sx - v.w / 2) / v.scale,
  v.cy - (sy - v.h / 2) / v.scale,
]

export function lerpTrack(track: [number, number][], stepH: number, tH: number): [number, number] {
  const f = Math.min(Math.max(tH / stepH, 0), track.length - 1)
  const k = Math.floor(f)
  if (k >= track.length - 1) return track[track.length - 1]
  const w = f - k
  return [track[k][0] * (1 - w) + track[k + 1][0] * w, track[k][1] * (1 - w) + track[k + 1][1] * w]
}

/** Position along a route (points carry cumulative t_h) at time tH. */
export function alongRoute(pts: { x: number; y: number; t_h: number }[], tH: number) {
  if (!pts.length) return null
  if (tH <= 0) return { x: pts[0].x, y: pts[0].y, i: 0 }
  for (let i = 1; i < pts.length; i++) {
    if (pts[i].t_h >= tH) {
      const a = pts[i - 1]
      const b = pts[i]
      const w = (tH - a.t_h) / Math.max(b.t_h - a.t_h, 1e-6)
      return { x: a.x + (b.x - a.x) * w, y: a.y + (b.y - a.y) * w, i }
    }
  }
  const last = pts[pts.length - 1]
  return { x: last.x, y: last.y, i: pts.length - 1 }
}
