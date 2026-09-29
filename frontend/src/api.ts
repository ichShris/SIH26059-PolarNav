// API client with an offline-first cache: every successful response is kept in
// localStorage, and when the shore link is down the dashboard runs from the last
// synced bundle.
import type { Coastline, GlobeLand, Meta, PolarisGrid, RoutePlan, SarResult, Scenario } from './types'

const PREFIX = 'polarnav:'

export interface Fetched<T> { data: T; offline: boolean; savedAt?: string }

function save(key: string, data: unknown) {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify({ savedAt: new Date().toISOString(), data }))
  } catch {
    // quota exceeded or storage blocked: the dashboard still works online
  }
}

function load<T>(key: string): { data: T; savedAt: string } | null {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

// Lets the operator simulate a satellite-link blackout from the Edge Sync panel.
let forceOffline = false
export const setForceOffline = (v: boolean) => {
  forceOffline = v
}

async function cached<T>(key: string, url: string, init?: RequestInit): Promise<Fetched<T>> {
  try {
    if (forceOffline) throw new TypeError('link down (simulated)')
    const r = await fetch(url, init)
    if (!r.ok) {
      const body = await r.json().catch(() => ({}))
      throw new ApiError(body.detail ?? `${r.status} ${r.statusText}`, r.status)
    }
    const data = (await r.json()) as T
    save(key, data)
    return { data, offline: false }
  } catch (e) {
    if (e instanceof ApiError) throw e
    const hit = load<T>(key)
    if (hit) return { data: hit.data, offline: true, savedAt: hit.savedAt }
    throw e
  }
}

export class ApiError extends Error {
  status: number
  constructor(msg: string, status: number) {
    super(msg)
    this.status = status
  }
}

export const api = {
  meta: () => cached<Meta>('meta', '/api/meta'),
  coastline: () => cached<Coastline>('coastline', '/api/coastline'),
  globeLand: () => cached<GlobeLand>('globe', '/api/coastline/globe'),
  scenario: (date: string) => cached<Scenario>(`scenario:last`, `/api/scenario?date=${date}`),
  polaris: (date: string, iceClass: string) =>
    cached<PolarisGrid>(`polaris:last`, `/api/polaris?date=${date}&ice_class=${encodeURIComponent(iceClass)}`),
  route: (body: object) =>
    cached<RoutePlan>('route:last', '/api/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  sar: (seed: number, pfa: number) => cached<SarResult>('sar:last', `/api/sar?seed=${seed}&pfa=${pfa}`),
  bundle: async (body: object) => {
    const r = await fetch('/api/sync/bundle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!r.ok) throw new ApiError('bundle download failed', r.status)
    return r.blob()
  },
}

export function decodeU8(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

export function decodeI8(b64: string): Int8Array {
  const u = decodeU8(b64)
  return new Int8Array(u.buffer, u.byteOffset, u.byteLength)
}
