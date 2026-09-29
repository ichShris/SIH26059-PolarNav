// Decoding and per-cell colouring of the ice-grid raster layers, shared by the polar chart and the globe.
import { decodeI8, decodeU8 } from './api'
import { diffColor, rioColor, sicColor, thickColor } from './colors'
import type { PolarisGrid, Scenario } from './types'

export type RasterLayer = 'sic' | 'thickness' | 'polaris' | 'error' | 'none'
export type RGBA = [number, number, number, number]

export interface Decoded {
  sic: Uint8Array[]
  truth: Uint8Array[]
  thick: Uint8Array[]
  rio?: Int8Array[]
  level?: Uint8Array[]
  live24?: Int8Array // yesterday's +24 h forecast minus today's analysis (%)
}

export function decodeScenario(s: Scenario, p: PolarisGrid | null): Decoded {
  return {
    sic: s.sic.map(decodeU8),
    truth: s.sic_truth.map(decodeU8),
    thick: s.thickness.map(decodeU8),
    rio: p?.rio.map(decodeI8),
    level: p?.level.map(decodeU8),
    live24: s.live_check?.error_24h ? decodeI8(s.live_check.error_24h) : undefined,
  }
}

/** Colour of ice-grid cell i (row-major, row 0 = south) for a layer and forecast slot, or null if transparent. */
export function cellColor(d: Decoded, layer: RasterLayer, slot: number, i: number): RGBA | null {
  const s = Math.min(slot, d.sic.length - 1)
  let c: RGBA
  if (layer === 'sic') c = sicColor(d.sic[s][i])
  else if (layer === 'thickness') c = thickColor(d.thick[s][i] / 50)
  else if (layer === 'polaris') c = d.rio && d.level ? rioColor(d.rio[s][i], d.level[s][i], d.sic[s][i] > 2) : [0, 0, 0, 0]
  else if (layer === 'error')
    c = s > 0 ? (d.truth[s - 1] ? diffColor(d.sic[s][i] - d.truth[s - 1][i]) : [0, 0, 0, 0]) : d.live24 ? diffColor(d.live24[i]) : [0, 0, 0, 0]
  else return null
  return c[3] < 8 ? null : c
}
