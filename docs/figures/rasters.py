"""Raster pieces of the figures (field maps, surface renders, animations), in the two-hue style of figstyle.tex.

Axes, color bars and labels are drawn by the TeX files; these functions only write images.
  signed fields (Re p)  : DIV  dark gray - white - navy, symmetric about zero
  levels, errors        : SEQ  white - navy
  the body (inside)     : INK
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


NAVY, GRAY, INK, BODY = _rgb("1F4E79"), _rgb("6B7280"), _rgb("1F2933"), _rgb("3E4752")
# keep in sync with the pgfplots colormaps in figstyle.tex
DIV_STOPS = ["2F3640", "8D95A0", "FFFFFF", "5E86AE", "12314F"]
SEQ_STOPS = ["FFFFFF", "C9D7E6", "7FA0C2", "1F4E79", "0F2A44"]
DIV = LinearSegmentedColormap.from_list("div", [_rgb(h) for h in DIV_STOPS])
SEQ = LinearSegmentedColormap.from_list("seq", [_rgb(h) for h in SEQ_STOPS])


def signed_rgb(field, vmax, gamma=0.8, body=BODY):
    """RGB image of Re(field) in [-vmax, vmax]; NaN (inside the body) painted with `body`."""
    re = np.real(field) / vmax
    s = np.sign(re) * np.clip(np.abs(re), 0, 1) ** gamma
    img = DIV((s + 1) / 2)[..., :3]
    img[~np.isfinite(field)] = body
    return img


def seq_rgb(values, lo, hi, body=BODY):
    img = SEQ(np.clip((values - lo) / (hi - lo), 0, 1))[..., :3]
    img[~np.isfinite(values)] = body
    return img


def _to_image(img, width=None, aspect=None, flip=True):
    a = np.clip(np.asarray(img) * 255, 0, 255).astype(np.uint8)
    im = Image.fromarray(a[::-1] if flip else a)
    if width is not None:                        # physical aspect (height / width) of the plotted region
        aspect = aspect if aspect is not None else im.height / im.width
        im = im.resize((int(width), int(round(width * aspect))), Image.BICUBIC)
    return im


def save_png(img, path, flip=True, width=None, aspect=None):
    """img rows along v (bottom row first, as plane_grid) -> PNG with the top row first; optionally resampled to
    `width` pixels and the physical aspect ratio height / width."""
    _to_image(img, width, aspect, flip).save(path)


def animation(field, vmax, path, frames=30, ms=60, width=480, aspect=None, gamma=0.8, canvas=None, colors=96,
              dither=False):
    """GIF of Re(field e^{j w t}) over one period (e^{+jwt} convention: the wave travels away from the source).

    canvas: if given, every frame is fitted into a white canvas x canvas square (aspect kept), so that GIFs of
    differently shaped regions line up. One palette for all frames (no color flicker between frames); dither
    hides banding in slow gradients at the cost of a larger file."""
    rgb = []
    for i in range(frames):
        f = field * np.exp(2j * np.pi * i / frames)
        im = _to_image(signed_rgb(f, vmax, gamma), width, aspect)
        if canvas:
            scale = canvas / max(im.size)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
            sq = Image.new("RGB", (canvas, canvas), (255, 255, 255))
            sq.paste(im, ((canvas - im.width) // 2, (canvas - im.height) // 2))
            im = sq
        rgb.append(im.convert("RGB"))
    # shared palette from a strip of evenly spaced frames
    sample = rgb[:: max(1, frames // 6)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * im.height))
    pal = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    imgs = [im.quantize(palette=pal, dither=mode) for im in rgb]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=True)


def render_surface(mesh, values, lo, hi, path, elev=25, azim=-60, size=900, light=(-0.4, -0.6, 0.7), cmap=None,
                   edges=None, edge_width=0.05):
    """Shaded render of the surface colored by `values` (per element, e.g. level in dB) with `cmap` (DIV by
    default: lo gray, middle white, hi navy). Lambert shading with a soft ambient term. Transparent background.
    edges: color of the triangle edges (default: same as the faces, i.e. hidden)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    col = (cmap or DIV)(np.clip((values - lo) / (hi - lo), 0, 1))[:, :3]
    e, a = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    lam = np.clip(mesh.normals @ L, 0, 1)
    shade = 0.62 + 0.38 * lam
    rgb = np.clip(col * shade[:, None], 0, 1)
    fig = plt.figure(figsize=(size / 200, size / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.add_collection3d(Poly3DCollection(mesh.verts[mesh.tris], facecolors=rgb,
                                         edgecolors=rgb if edges is None else edges, linewidths=edge_width))
    lo3, hi3 = mesh.verts.min(0), mesh.verts.max(0)
    c, r = (lo3 + hi3) / 2, (hi3 - lo3).max() / 2 * 0.82
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.set_proj_type("persp", focal_length=0.6)
    ax.set_axis_off()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def render_with_plane(mesh, face_rgb, plane_pts, plane_rgb, path, elev=25, azim=-60, size=900,
                      light=(-0.4, -0.6, 0.7), step=2):
    """Shaded body (face colors face_rgb (N, 3)) standing in a colored plane: plane_pts (n, m, 3) grid points,
    plane_rgb (n, m, 3) colors, NaN points (inside the body) left out. Body and plane quads go into one collection
    so that matplotlib sorts them by depth together (the plane cuts through the body). Transparent background."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    shade = 0.62 + 0.38 * np.clip(mesh.normals @ L, 0, 1)
    polys = list(mesh.verts[mesh.tris])
    cols = list(np.clip(face_rgb * shade[:, None], 0, 1))
    P, C = plane_pts[::step, ::step], plane_rgb[::step, ::step]
    ok = np.isfinite(P).all(axis=2)
    for i in range(P.shape[0] - 1):
        for j in range(P.shape[1] - 1):
            if ok[i, j] and ok[i + 1, j] and ok[i, j + 1] and ok[i + 1, j + 1]:
                polys.append(np.array([P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]]))
                cols.append(C[i, j])
    fig = plt.figure(figsize=(size / 200, size / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.3))   # no seams
    allp = np.concatenate([mesh.verts, P[ok]])
    lo3, hi3 = allp.min(0), allp.max(0)
    c, r = (lo3 + hi3) / 2, (hi3 - lo3).max() / 2 * 0.82
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.set_proj_type("persp", focal_length=0.6)
    ax.set_axis_off()
    fig.savefig(path, transparent=True)
    plt.close(fig)
