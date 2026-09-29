// GPU renderer for the globe's filled layers (space, ocean, sea ice, land, shading).
//
// One full-screen quad; the fragment shader inverts the orthographic projection for every
// pixel and samples three textures:
//   uWorld  equirectangular land colours for the whole Earth (4096 x 2048)
//   uPolar  polar-stereographic land colours for Antarctica (2048^2, ~4 km/px, sharper than uWorld there)
//   uIce    the 160 x 160 sea-ice / risk raster of the active layer (bilinear filtered)
// Cost is per pixel, independent of how many polygons or cells are visible, so rotation and
// deep zoom stay at display frame rate.
import { TWO_RK } from './geo'

const VERT = `
attribute vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
`

const FRAG = `
precision highp float;
uniform vec2 uCenter;
uniform float uR;
uniform vec3 uE;
uniform vec3 uN;
uniform vec3 uC;
uniform sampler2D uWorld;
uniform sampler2D uPolar;
uniform sampler2D uIce;
uniform float uHasIce;
uniform float uPolarHalf;
uniform float uIceHalf;
const float PI = 3.14159265358979;
const float TWO_RK = ${TWO_RK.toFixed(4)};

bool inUnit(vec2 uv) { return uv.x >= 0.0 && uv.y >= 0.0 && uv.x <= 1.0 && uv.y <= 1.0; }

void main() {
  vec2 p = (gl_FragCoord.xy - uCenter) / uR;
  float r2 = dot(p, p);
  if (r2 > 1.0) {
    float d = sqrt(r2);
    float halo = clamp(1.0 - (d - 1.0) / 0.12, 0.0, 1.0);
    gl_FragColor = vec4(mix(vec3(0.012, 0.039, 0.075), vec3(0.37, 0.83, 1.0), 0.35 * halo * halo), 1.0);
    return;
  }
  float z = sqrt(1.0 - r2);
  vec3 v = p.x * uE + p.y * uN + z * uC;
  float lat = asin(clamp(v.z, -1.0, 1.0));
  float lon = atan(v.y, v.x);

  // ocean, lit from the upper left
  float g = clamp(length(p - vec2(-0.35, 0.35)) / 1.25, 0.0, 1.0);
  vec3 col = mix(vec3(0.106, 0.290, 0.451), vec3(0.027, 0.102, 0.176), g);

  // polar stereographic position (km) of this pixel, used by the ice and Antarctic textures
  vec2 xy = vec2(0.0);
  if (lat < -0.5) {
    float rho = TWO_RK * tan(PI / 4.0 + lat / 2.0);
    xy = vec2(rho * sin(lon), rho * cos(lon));
  }

  if (uHasIce > 0.5 && lat < -0.5) {
    vec2 iuv = (xy + uIceHalf) / (2.0 * uIceHalf);
    if (inUnit(iuv)) {
      vec4 ice = texture2D(uIce, iuv);     // premultiplied alpha
      col = ice.rgb + col * (1.0 - ice.a);
    }
  }

  vec4 land;
  vec2 puv = (xy + uPolarHalf) / (2.0 * uPolarHalf);
  if (lat < -0.5 && inUnit(puv)) land = texture2D(uPolar, puv);
  else land = texture2D(uWorld, vec2((lon + PI) / (2.0 * PI), (lat + PI / 2.0) / PI));
  col = land.rgb + col * (1.0 - land.a);

  // limb darkening and a thin atmospheric rim
  float sh = smoothstep(0.2, 1.02, length(p - vec2(-0.3, 0.3)));
  col *= 1.0 - 0.45 * sh * sh;
  col = mix(col, vec3(0.37, 0.83, 1.0), 0.6 * smoothstep(0.985, 1.0, sqrt(r2)));
  gl_FragColor = vec4(col, 1.0);
}
`

export interface GlobeUniforms {
  cx: number
  cy: number
  R: number
  e: [number, number, number]
  n: [number, number, number]
  c: [number, number, number]
}

type Ring = [number, number][]

export const LAND_COLOR = '#2b3f33'
export const ANTARCTIC_COLOR = '#dfeaf2'
export const SHELF_COLOR = '#c3d6e4'

/**
 * Draw lon/lat rings onto an equirectangular canvas. Longitudes are unwrapped so rings that
 * cross the antimeridian stay continuous; a ring that circles a pole (net longitude change of
 * 360 deg, i.e. Antarctica) is closed along that pole. Each ring is painted at -360/0/+360
 * offsets so wrapped parts land on the canvas.
 */
function paintEquirect(ctx: CanvasRenderingContext2D, W: number, H: number, polys: Ring[][], color: string) {
  const X = (lon: number) => ((lon + 180) / 360) * W
  const Y = (lat: number) => ((90 - lat) / 180) * H
  ctx.fillStyle = color
  for (const poly of polys) {
    for (const off of [-360, 0, 360]) {
      ctx.beginPath()
      for (const ring of poly) {
        const pts: [number, number][] = []
        let prev = ring[0][0]
        let acc = ring[0][0]
        for (const [lon, lat] of ring) {
          let d = lon - prev
          if (d > 180) d -= 360
          else if (d < -180) d += 360
          acc += d
          prev = lon
          pts.push([acc, lat])
        }
        pts.forEach(([lon, lat], i) => (i === 0 ? ctx.moveTo(X(lon + off), Y(lat)) : ctx.lineTo(X(lon + off), Y(lat))))
        const net = pts[pts.length - 1][0] - pts[0][0]
        if (Math.abs(net) > 180) {
          const poleLat = ring.reduce((s, q) => s + q[1], 0) / ring.length < 0 ? -90 : 90
          ctx.lineTo(X(pts[pts.length - 1][0] + off), Y(poleLat))
          ctx.lineTo(X(pts[0][0] + off), Y(poleLat))
        }
        ctx.closePath()
      }
      ctx.fill('evenodd')
    }
  }
}

export function buildWorldCanvas(land: Ring[][], shelf: Ring[][], maxTex: number): HTMLCanvasElement {
  const W = Math.min(4096, maxTex)
  const H = W / 2
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const ctx = c.getContext('2d')!
  const isAnt = (p: Ring[]) => p[0].some(([, lat]) => lat < -65)
  paintEquirect(ctx, W, H, land.filter((p) => !isAnt(p)), LAND_COLOR)
  paintEquirect(ctx, W, H, land.filter(isAnt), ANTARCTIC_COLOR)
  paintEquirect(ctx, W, H, shelf, SHELF_COLOR)
  return c
}

/** Polar-stereographic land/shelf texture from the projected-km coastline used by the polar chart. */
export function buildPolarCanvas(land: [number, number][][], shelf: [number, number][][], halfKm: number, size: number): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = size
  c.height = size
  const ctx = c.getContext('2d')!
  const S = (x: number, y: number): [number, number] => [((x + halfKm) / (2 * halfKm)) * size, ((halfKm - y) / (2 * halfKm)) * size]
  const fill = (rings: [number, number][][], color: string, pred: (r: [number, number][]) => boolean) => {
    ctx.beginPath()
    for (const r of rings) {
      if (!pred(r)) continue
      r.forEach(([x, y], i) => {
        const [sx, sy] = S(x, y)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.closePath()
    }
    ctx.fillStyle = color
    ctx.fill('evenodd')
  }
  // Antarctica sits around the pole (small projected radius); everything else is other land
  const polar = (r: [number, number][]) => r.some(([x, y]) => Math.hypot(x, y) < 2600)
  fill(land, LAND_COLOR, (r) => !polar(r))
  fill(land, ANTARCTIC_COLOR, polar)
  fill(shelf, SHELF_COLOR, () => true)
  return c
}

export class GlobeGL {
  private gl: WebGLRenderingContext
  private prog: WebGLProgram
  private loc: Record<string, WebGLUniformLocation | null> = {}
  private tex: Record<'world' | 'polar' | 'ice', WebGLTexture>
  private hasIce = false
  maxTex: number

  static create(canvas: HTMLCanvasElement): GlobeGL | null {
    const gl = canvas.getContext('webgl', { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: false })
    return gl ? new GlobeGL(gl) : null
  }

  private constructor(gl: WebGLRenderingContext) {
    this.gl = gl
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!
      gl.shaderSource(s, src)
      gl.compileShader(s)
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader error')
      return s
    }
    const prog = gl.createProgram()!
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT))
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG))
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) ?? 'link error')
    this.prog = prog
    gl.useProgram(prog)
    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)
    const aPos = gl.getAttribLocation(prog, 'aPos')
    gl.enableVertexAttribArray(aPos)
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0)
    for (const u of ['uCenter', 'uR', 'uE', 'uN', 'uC', 'uWorld', 'uPolar', 'uIce', 'uHasIce', 'uPolarHalf', 'uIceHalf']) {
      this.loc[u] = gl.getUniformLocation(prog, u)
    }
    this.tex = { world: gl.createTexture()!, polar: gl.createTexture()!, ice: gl.createTexture()! }
    // start with 1x1 transparent textures so the shader always has valid samplers
    for (const t of Object.values(this.tex)) {
      gl.bindTexture(gl.TEXTURE_2D, t)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4))
    }
    gl.uniform1i(this.loc.uWorld, 0)
    gl.uniform1i(this.loc.uPolar, 1)
    gl.uniform1i(this.loc.uIce, 2)
  }

  private upload(which: 'world' | 'polar', canvas: HTMLCanvasElement, repeatS: boolean) {
    const gl = this.gl
    gl.bindTexture(gl.TEXTURE_2D, this.tex[which])
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true) // canvas row 0 is north / max y; texture v=0 is south
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas)
    gl.generateMipmap(gl.TEXTURE_2D)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, repeatS ? gl.REPEAT : gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  }

  setWorld(canvas: HTMLCanvasElement) {
    this.upload('world', canvas, true)
  }

  setPolar(canvas: HTMLCanvasElement) {
    this.upload('polar', canvas, false)
  }

  /** rgba: n*n*4 straight-alpha bytes, row 0 = southern edge (min y). null clears the layer. */
  setIce(rgba: Uint8Array | null, n: number) {
    const gl = this.gl
    this.hasIce = !!rgba
    if (!rgba) return
    const pm = new Uint8Array(rgba.length)
    for (let i = 0; i < rgba.length; i += 4) {
      const a = rgba[i + 3] / 255
      pm[i] = rgba[i] * a
      pm[i + 1] = rgba[i + 1] * a
      pm[i + 2] = rgba[i + 2] * a
      pm[i + 3] = rgba[i + 3]
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex.ice)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, n, n, 0, gl.RGBA, gl.UNSIGNED_BYTE, pm)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  }

  render(width: number, height: number, u: GlobeUniforms, polarHalfKm: number, iceHalfKm: number) {
    const gl = this.gl
    gl.viewport(0, 0, width, height)
    gl.useProgram(this.prog)
    gl.uniform2f(this.loc.uCenter, u.cx, u.cy)
    gl.uniform1f(this.loc.uR, u.R)
    gl.uniform3fv(this.loc.uE, u.e)
    gl.uniform3fv(this.loc.uN, u.n)
    gl.uniform3fv(this.loc.uC, u.c)
    gl.uniform1f(this.loc.uHasIce, this.hasIce ? 1 : 0)
    gl.uniform1f(this.loc.uPolarHalf, polarHalfKm)
    gl.uniform1f(this.loc.uIceHalf, iceHalfKm)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.tex.world)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.tex.polar)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, this.tex.ice)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
  }

  dispose() {
    const gl = this.gl
    for (const t of Object.values(this.tex)) gl.deleteTexture(t)
    gl.deleteProgram(this.prog)
  }
}
