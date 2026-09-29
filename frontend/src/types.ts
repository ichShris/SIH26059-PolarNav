export interface Place { name: string; lat: number; lon: number; kind: 'port' | 'station' | 'vessel' }

export interface IceClassInfo { name: string; polar_class: boolean; capability_m: number; elevated_speed_kn: number }

export interface Skill { rmse: number; iiee: number }

export interface SicMetrics {
  params: number
  iterations: number
  train_minutes: number
  splits: Record<string, [string, string]>
  val: Record<'model' | 'persistence' | 'climatology', Skill[]>
  test: Record<'model' | 'persistence' | 'climatology', Skill[]>
  leads_h: number[]
}

export interface DriftMetrics {
  n_tracks: number
  horizon_h: number
  persistence_km: { mean: number; median: number }
  physics_km: { mean: number; median: number }
  hybrid_km: { mean: number; median: number }
}

export interface Meta {
  grid: { cell_km: number; nav_n: number; nav_half_km: number; ice_n: number; ice_off: number }
  dates: { min: string; max: string; default: string; test_from: string }
  ice_classes: IceClassInfo[]
  places: Record<string, Place>
  vessel: Record<string, number | string>
  polaris: { ice_types: string[]; riv: Record<string, number[]> }
  metrics: { sic: SicMetrics | null; drift: DriftMetrics | null; routes: RouteBenchmark | null; sar: { scenes: number; pfa: number; recall: number; precision: number } }
  data_mode: string
}

export interface Ellipse { t_h: number; x: number; y: number; a: number; b: number; angle: number }

export interface Berg {
  id: string
  lat: number
  lon: number
  x: number
  y: number
  length_m: number
  width_m: number
  height_m: number
  draft_m: number
  observed: [number, number][]
  forecast: [number, number][]
  physics_only: [number, number][]
  truth: [number, number][]
  members_72h: [number, number][]
  ellipses: Ellipse[]
  error_72h_km: { hybrid: number; physics: number }
  grounded: boolean
}

export interface Scenario {
  date: string
  day: number
  is_test_period: boolean
  sic: string[]
  sic_truth: string[]
  thickness: string[]
  extent_km2: number[]
  wind: { n: number; step_km: number; u: number[]; v: number[] }
  bergs: { step_h: number; members: number; bergs: Berg[]; summary: { mean_err_72h_km: { hybrid: number; physics: number } } }
  skill: { lead_h: number; model: Skill; persistence: Skill }[]
  live_check: LiveCheck
}

export interface PolarisGrid { ice_class: string; rio: string[]; level: string[] }

export interface RoutePoint {
  x: number; y: number; lat: number; lon: number; t_h: number; sic: number; thick_m: number
  rio: number; level: number; speed_kn: number; fuel_t: number; dist_nm: number
}

export interface Waypoint {
  name: string; lat: number; lon: number; x: number; y: number; t_h: number; sic: number; rio: number
  speed_kn: number; dist_nm: number; fuel_t: number; course: number | null
}

export interface RouteSummary {
  distance_nm: number; hours: number; days: number; fuel_t: number; co2_t: number; min_rio: number
  worst_level: string; elevated_nm: number; ice_nm: number; beyond_forecast_horizon: boolean
  min_berg_cpa_km: number | null; min_berg_cpa_truth_km: number | null; bergs_within_10nm: number
}

export interface RouteEval {
  points: RoutePoint[]
  waypoints: Waypoint[]
  summary: RouteSummary
  berg_clearance: { id: string; cpa_km: number; t_h: number; cpa_truth_km: number }[]
}

export interface SyncStats {
  raw_bytes: number
  packed_bytes: number
  ratio: number
  contents: Record<string, number>
  transfer_s: Record<string, { raw: number; packed: number }>
}

export interface RouteAlternative extends RouteEval { key: string; label: string; same_track_as_optimal: boolean }

export interface LiveCheck {
  per_lead: { lead_h: number; issued: string; model: Skill; persistence: Skill }[]
  error_24h: string | null
  reference: string
}

export interface RoutePlan {
  request: Record<string, unknown>
  warnings: string[]
  optimal: RouteEval | null
  baseline: RouteEval | null
  alternatives: RouteAlternative[]
  comparison?: { fuel_saving_t: number; fuel_saving_pct: number; co2_saving_t: number; time_delta_h: number; distance_delta_nm: number }
  compute_s: number
  sync: SyncStats
}

export interface SarResult {
  seed: number; size: number; pixel_m: number; pfa: number; image_b64: string
  detections: { x0: number; y0: number; x1: number; y1: number; area_m2: number; length_m: number; peak_db: number; match: boolean }[]
  truth: { x0: number; y0: number; x1: number; y1: number; length_m: number; in_pack: boolean }[]
  scores: { recall: number; precision: number; n_truth: number; n_detections: number }
}

export interface Coastline { land: [number, number][][]; shelf: [number, number][][] }

/** lon/lat polygons (rings of [lon, lat]) for the globe */
export interface GlobeLand {
  land: [number, number][][][]
  shelf: [number, number][][][]
  land_lo?: [number, number][][][] // low-detail outline used while the globe moves
  shelf_lo?: [number, number][][][]
}

export interface RouteForm {
  origin: string
  dest: string
  originLL?: { lat: number; lon: number }
  destLL?: { lat: number; lon: number }
  cruise_kn: number
  w_time: number
  w_risk: number
  avoid_bergs: boolean
  w_ice: number
  berg_margin_nm: number
}

export interface RouteBenchmark {
  ice_class: string
  voyages: number
  feasible: number
  fuel_saving_pct: { mean: number; median: number; p10: number; p90: number; max: number }
  fuel_saving_pct_heavy_ice: { n: number; mean: number | null; median: number | null }
  mean_time_saved_h: number
  voyages_passing_berg_within_10nm: { polarnav: number; conventional: number }
}
