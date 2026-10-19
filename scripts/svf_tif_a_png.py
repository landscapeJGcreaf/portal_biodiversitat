#!/usr/bin/env python3
"""
Variant de tif_a_png.py que llegeix el GeoTIFF amb Pillow en lloc de
tifffile, perquè admet TIFs amb compressió JPEG sense necessitar el
paquet 'imagecodecs'. La resta de la lògica (reprojecció a Web Mercator,
graella de sortida, colorització, JSON de consulta) és idèntica.
"""
import argparse, json, math, sys
import numpy as np
from PIL import Image

A, F, K0, LON0, E0 = 6378137.0, 1 / 298.257222101, 0.9996, math.radians(3.0), 500000.0
N_ = F / (2 - F)
AA = A / (1 + N_) * (1 + N_**2 / 4 + N_**4 / 64)
ALFA = [N_/2 - 2*N_**2/3 + 5*N_**3/16, 13*N_**2/48 - 3*N_**3/5, 61*N_**3/240]
BETA = [N_/2 - 2*N_**2/3 + 37*N_**3/96, N_**2/48 + N_**3/15, 17*N_**3/480]
DELTA = [2*N_ - 2*N_**2/3 - 2*N_**3, 7*N_**2/3 - 8*N_**3/5, 56*N_**3/15]
R = 6378137.0


def directa(lon, lat):
    lam, phi = np.radians(lon) - LON0, np.radians(lat)
    e = 2 * math.sqrt(N_) / (1 + N_)
    t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi_, eta_ = np.arctan2(t, np.cos(lam)), np.arctanh(np.sin(lam) / np.sqrt(1 + t**2))
    xi = xi_ + sum(a * np.sin(2*j*xi_) * np.cosh(2*j*eta_) for j, a in enumerate(ALFA, 1))
    eta = eta_ + sum(a * np.cos(2*j*xi_) * np.sinh(2*j*eta_) for j, a in enumerate(ALFA, 1))
    return E0 + K0 * AA * eta, K0 * AA * xi


def inversa(E, N):
    xi, eta = N / (K0 * AA), (E - E0) / (K0 * AA)
    xi_ = xi - sum(b * np.sin(2*j*xi) * np.cosh(2*j*eta) for j, b in enumerate(BETA, 1))
    eta_ = eta - sum(b * np.cos(2*j*xi) * np.sinh(2*j*eta) for j, b in enumerate(BETA, 1))
    chi = np.arcsin(np.sin(xi_) / np.cosh(eta_))
    phi = chi + sum(d * np.sin(2*j*chi) for j, d in enumerate(DELTA, 1))
    lam = np.arctan2(np.sinh(eta_), np.cos(xi_))
    return np.degrees(LON0 + lam), np.degrees(phi)


def merc(lon, lat):
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def merc_inv(x, y):
    return np.degrees(x / R), np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)


def llegeix_tif(cami):
    im = Image.open(cami)
    tags = im.tag_v2
    dades = np.array(im).astype("float64")
    if dades.ndim != 2:
        sys.exit("El TIF ha de tenir una sola banda.")
    gk = tags.get(34735, ())
    claus = {gk[i]: gk[i + 3] for i in range(4, len(gk), 4)}
    epsg = claus.get(3072) or claus.get(2048)
    if epsg not in (25831, 4326):
        sys.exit(f"Aquest script només admet EPSG:25831 o EPSG:4326 (el fitxer és {epsg}).")
    px, py, _ = tags[33550]
    tx, ty = tags[33922][3:5]
    nodata = tags.get(42113)
    return dades, tx, ty, px, py, epsg, (float(nodata) if nodata is not None else None)


def transformacions(epsg, x0, y0, px, py):
    if epsg == 25831:
        def lonlat_a_pixel(lon, lat):
            E, N = directa(lon, lat)
            return (E - x0) / px, (y0 - N) / py

        def pixel_a_lonlat(col, fila):
            return inversa(x0 + col * px, y0 - fila * py)
    else:
        def lonlat_a_pixel(lon, lat):
            return (lon - x0) / px, (y0 - lat) / py

        def pixel_a_lonlat(col, fila):
            return x0 + col * px, y0 - fila * py
    return lonlat_a_pixel, pixel_a_lonlat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif"); ap.add_argument("png")
    ap.add_argument("--parades", type=float, nargs="+", required=True)
    ap.add_argument("--colors", required=True)
    ap.add_argument("--res", type=float, default=5)
    ap.add_argument("--nodata", type=float, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--json-stride", type=int, default=1,
                     help="sobremostreja la graella del --json cada N píxels de la font (per no generar un JSON gegant en TIFs molt detallats)")
    a = ap.parse_args()

    cols = [c.strip().lstrip("#") for c in a.colors.split(",")]
    if len(cols) != len(a.parades):
        sys.exit("Cal un color per cada parada.")
    rgb = np.array([[int(c[i:i + 2], 16) for i in (0, 2, 4)] for c in cols], dtype=float)

    dades, x0, y0, px, py, epsg, nd_tif = llegeix_tif(a.tif)
    nodata = a.nodata if a.nodata is not None else nd_tif
    h, w = dades.shape
    lonlat_a_pixel, pixel_a_lonlat = transformacions(epsg, x0, y0, px, py)

    if epsg == 25831:
        lo, la = pixel_a_lonlat(np.array([0.0]), np.array([0.0]))
        c2, f2 = lonlat_a_pixel(lo, la)
        assert abs(c2[0]) < 1e-3 and abs(f2[0]) < 1e-3, "Error en la projecció"

    cc = np.array([0.0, w, w, 0.0]); cf = np.array([0.0, 0.0, h, h])
    clon, clat = pixel_a_lonlat(cc, cf)
    lon_min, lon_max, lat_min, lat_max = clon.min(), clon.max(), clat.min(), clat.max()
    xm0, ym1 = merc(lon_min, lat_max); xm1, ym0 = merc(lon_max, lat_min)

    pm = a.res / math.cos(math.radians((lat_min + lat_max) / 2))
    W, H = math.ceil((xm1 - xm0) / pm), math.ceil((ym1 - ym0) / pm)
    xs = xm0 + (np.arange(W) + 0.5) * pm
    ys = ym1 - (np.arange(H) + 0.5) * pm
    X, Y = np.meshgrid(xs, ys)
    lon, lat = merc_inv(X, Y)
    colf, filaf = lonlat_a_pixel(lon, lat)
    col, fila = np.floor(colf).astype(int), np.floor(filaf).astype(int)
    dins = (col >= 0) & (col < w) & (fila >= 0) & (fila < h)
    val = np.full((H, W), np.nan)
    val[dins] = dades[fila[dins], col[dins]]
    valid = dins & np.isfinite(val)
    if nodata is not None:
        valid &= val != nodata

    out = np.zeros((H, W, 4), dtype=np.uint8)
    v = val[valid]
    for k in range(3):
        out[..., k][valid] = np.interp(v, a.parades, rgb[:, k]).round().astype(np.uint8)
    out[..., 3][valid] = 255
    Image.fromarray(out, "RGBA").save(a.png, optimize=True)

    if a.json:
        s = a.json_stride
        dj = dades[::s, ::s]
        hj, wj = dj.shape
        ok = np.isfinite(dj)
        if nodata is not None:
            ok &= dj != nodata
        idx = np.flatnonzero(ok.ravel())
        graella = {
            "amplada": wj, "alcada": hj, "x0": x0, "y0": y0, "px": px * s, "py": py * s, "epsg": epsg,
            "i": idx.tolist(),
            "v": [round(float(x), 4) for x in dj.ravel()[idx]],
        }
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(graella, fh, separators=(",", ":"))

    dades_valides = dades[(dades != nodata) & np.isfinite(dades)] if nodata is not None else dades[np.isfinite(dades)]
    res = {
        "epsg_entrada": epsg,
        "coordenades": [[lon_min, lat_max], [lon_max, lat_max], [lon_max, lat_min], [lon_min, lat_min]],
        "mida_png": [W, H],
        "pixels_amb_dada_origen": int(dades_valides.size),
        "pixels_amb_dada_png": int(valid.sum()),
        "valors": [float(dades_valides.min()), float(dades_valides.max())] if dades_valides.size else None,
    }
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
