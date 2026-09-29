// Colour ramps for raster layers. All return [r, g, b, a] with a in 0..255.
type RGBA = [number, number, number, number]

function ramp(stops: [number, [number, number, number]][], t: number): [number, number, number] {
  if (t <= stops[0][0]) return stops[0][1]
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1]
      const [t1, c1] = stops[i]
      const w = (t - t0) / (t1 - t0)
      return [c0[0] + (c1[0] - c0[0]) * w, c0[1] + (c1[1] - c0[1]) * w, c0[2] + (c1[2] - c0[2]) * w]
    }
  }
  return stops[stops.length - 1][1]
}

const SIC_STOPS: [number, [number, number, number]][] = [
  [0.15, [38, 84, 140]],
  [0.4, [86, 148, 205]],
  [0.7, [168, 208, 238]],
  [0.9, [226, 242, 252]],
  [1.0, [250, 253, 255]],
]

/** Sea-ice concentration 0..100 */
export function sicColor(pct: number): RGBA {
  const t = pct / 100
  if (t < 0.02) return [0, 0, 0, 0]
  if (t < 0.15) return [38, 84, 140, Math.round(40 + 600 * t)]
  const [r, g, b] = ramp(SIC_STOPS, t)
  return [r, g, b, 235]
}

const THICK_STOPS: [number, [number, number, number]][] = [
  [0.0, [45, 70, 120]],
  [0.5, [90, 120, 200]],
  [1.0, [150, 120, 220]],
  [2.0, [215, 110, 200]],
  [3.0, [255, 170, 150]],
  [4.5, [255, 235, 200]],
]

export function thickColor(m: number): RGBA {
  if (m < 0.03) return [0, 0, 0, 0]
  const [r, g, b] = ramp(THICK_STOPS, m)
  return [r, g, b, 225]
}

/** POLARIS: colour by RIO; green >= 0, amber elevated, red special consideration. */
export function rioColor(rio: number, level: number, hasIce: boolean): RGBA {
  if (!hasIce) return [0, 0, 0, 0]
  if (level === 2) return [230, 57, 70, 220]
  if (level === 1) {
    const t = Math.min(-rio / 10, 1)
    return [245, 170 - 60 * t, 40, 200]
  }
  const t = Math.min(rio / 30, 1)
  return [60 + 20 * (1 - t), 190 - 40 * (1 - t), 120, 110 + 60 * (1 - t)]
}

export function diffColor(d: number): RGBA {
  // forecast minus truth, percent SIC
  if (Math.abs(d) < 3) return [0, 0, 0, 0]
  const t = Math.min(Math.abs(d) / 40, 1)
  return d > 0 ? [240, 90, 90, 60 + 180 * t] : [80, 160, 255, 60 + 180 * t]
}

export const LEVEL_COLORS = ['#3ec28f', '#f5a623', '#e63946']
export const LEVEL_SHORT = ['Normal', 'Elevated', 'Special']
