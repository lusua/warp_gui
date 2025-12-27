# -*- coding: utf-8 -*-
"""
warp_image_gui.py

概要
----
Tkinter GUI 上で指定した一般多角形領域に、入力画像（矩形）をワープして貼り付けるツールです。
処理は「制約付き三角形分割（triangle） → 調和写像（cotangent Laplacian）でUV決定 →
三角形ごとの逆写像ラスタライズ（バイリニアサンプリング）」で行います。

主な機能
--------
- 画像の読み込み（表示はウィンドウに合わせて縮小。計算は常に元画像座標）
- 多角形編集
  - 頂点追加（シングルクリック：ダブルクリックと競合しないよう遅延確定）
  - 頂点削除（頂点上ダブルクリック）
  - 頂点挿入（線分上ダブルクリック：線分への射影点を挿入）
  - 頂点ドラッグ移動
  - 線分ドラッグ（線分の両端頂点を同じΔで平行移動）
  - 多角形全体ドラッグ（閉じた多角形内部をドラッグで全頂点を平行移動）
  - ポリゴン未確定のまま「ワープ実行」した場合は自動で閉路化（最後→最初を接続）
- 境界条件（境界UV固定）のモード切替
  - 原点(0,0)最近点
  - 4点（四隅）最近点
  - 8点（四隅＋辺中点）最近点
  - 放射（一定角度）交点対応：中心からの半直線と境界の交点を対応（複数交点は最遠を採用）
- 可視化
  - メッシュ表示（境界太線＋境界頂点点）／対応点（アンカー）表示
  - 入力画像上のUVメッシュ可視化（必要に応じて）
- 入出力
  - ワープ結果 PNG 保存
  - メッシュ可視化 PNG 保存
  - 多角形座標の保存／読込（JSON または簡易テキスト）
    * JSON には image_size/closed/vertices 等を記録し、読込時にサイズ差があれば自動スケーリング

必要要件
--------
- Python 3.9+ 推奨
- numpy, pillow, scipy
- triangle（drufat/triangle）
  pip install git+https://github.com/drufat/triangle

注意事項
--------
- 多角形は自己交差しない（単純多角形）ことを前提とします。
- メッシュ密度（max_area）を下げるほど品質は上がりますが計算時間が増えます。
- 境界条件は歪み品質に強く影響します。形状に応じてモード／放射ステップ等を調整してください。

ライセンス
----------
このファイルを含むリポジトリの LICENSE を参照してください。
"""
import math
import traceback
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
from PIL import Image, ImageTk, ImageDraw
import triangle as tr
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import tkinter as tk
from tkinter import filedialog, messagebox

import time


# ============================================================
#  幾何 / パラメタ化（調和写像） / ラスタライズ
# ============================================================

def polygon_signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if polygon_signed_area(poly) > 0 else poly[::-1].copy()

def build_pslg(poly: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(poly)
    segments = np.stack([np.arange(n), np.roll(np.arange(n), -1)], axis=1).astype(np.int32)
    return {"vertices": poly.astype(np.float64), "segments": segments}

def segments_boundary_order(vertices: np.ndarray, segments: np.ndarray) -> List[int]:
    """
    triangle 出力の segments は境界が細分化されることがある。
    それでも（外周のみなら）次数2の閉路になるので辿って順序化する。
    """
    adj: Dict[int, List[int]] = {}
    for a, b in segments:
        a = int(a); b = int(b)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    start = min(adj.keys())
    order = [start]
    prev = -1
    cur = start
    while True:
        nbrs = adj[cur]
        if len(nbrs) < 2:
            raise RuntimeError("Boundary is not a single cycle (vertex degree < 2).")
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(adj) + 5:
            raise RuntimeError("Boundary traversal failed (not a simple single cycle).")
    return order

def cumulative_arclength(points: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(points[1:] - points[:-1], axis=1)
    s = np.zeros(len(points), dtype=np.float64)
    s[1:] = np.cumsum(d)
    return s


def closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """点pから線分abへの最近点qと射影パラメータt(0..1)を返す。"""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return a.copy(), 0.0
    t = float(np.dot(p - a, ab) / denom)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    q = a + t * ab
    return q, t


def insert_closest_boundary_point_to_origin(poly: np.ndarray, origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """
    原点(0,0)に対して多角形境界上の最近点が「辺の内部」にある場合、
    その点を頂点として poly に挿入して返す。
    - poly: (n,2) 反時計回り/時計回りどちらでもよい（順序は保持）
    - 既に最近点が頂点に一致する場合はそのまま返す。
    """
    n = len(poly)
    if n < 2:
        return poly

    p = np.array(origin, dtype=np.float64)
    best_d2 = float("inf")
    best_i = 0
    best_q = None
    best_t = 0.0

    for i in range(n):
        j = (i + 1) % n
        a = poly[i]
        b = poly[j]
        q, t = closest_point_on_segment(p, a, b)
        d2 = float(np.dot(q - p, q - p))
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
            best_q = q
            best_t = t

    if best_q is None:
        return poly

    eps = 1e-6
    # 辺の内部なら挿入（端点近傍は「既に頂点とみなす」）
    if eps < best_t < (1.0 - eps):
        return np.insert(poly, best_i + 1, best_q, axis=0)
    return poly



def closest_point_on_polyline(p: np.ndarray, polyline: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    閉路 polyline（(n,2)）の各線分に対して点pの最近点qを求め、
    周回に沿った弧長パラメータ s（0..L）も返す。
    - polyline は「閉じていない」頂点列を想定（最後→最初の線分も含む）。
    """
    n = len(polyline)
    if n < 2:
        return polyline[0].copy(), 0.0

    best_d2 = float("inf")
    best_q = polyline[0].copy()
    best_s = 0.0

    # 事前に各辺長と累積長
    lens = []
    for i in range(n):
        j = (i + 1) % n
        d = float(np.linalg.norm(polyline[j] - polyline[i]))
        lens.append(d)
    cum = np.zeros(n + 1, dtype=np.float64)
    cum[1:] = np.cumsum(lens)

    for i in range(n):
        j = (i + 1) % n
        a = polyline[i]
        b = polyline[j]
        q, t = closest_point_on_segment(p, a, b)
        d2 = float(np.dot(q - p, q - p))
        if d2 < best_d2:
            best_d2 = d2
            best_q = q
            best_s = float(cum[i] + t * lens[i])
    # best_s は [0, L) の範囲
    L = float(cum[-1])
    if L > 1e-12:
        best_s = best_s % L
    return best_q, best_s


def insert_closest_boundary_points(poly: np.ndarray, targets: List[Tuple[float, float]]) -> np.ndarray:
    """
    targets（キャンバス座標）それぞれに対し、境界上の最近点が辺内部なら頂点挿入する。
    複数点を順に挿入するので、poly の頂点数は増える。
    """
    out = np.asarray(poly, dtype=np.float64)
    for tx, ty in targets:
        n = len(out)
        if n < 2:
            continue

        p = np.array([tx, ty], dtype=np.float64)
        best_d2 = float("inf")
        best_i = 0
        best_q = None
        best_t = 0.0

        for i in range(n):
            j = (i + 1) % n
            a = out[i]
            b = out[j]
            q, t = closest_point_on_segment(p, a, b)
            d2 = float(np.dot(q - p, q - p))
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_q = q
                best_t = t

        if best_q is None:
            continue

        # 既に既存頂点とほぼ同一点なら挿入しない
        if np.min(np.sum((out - best_q) ** 2, axis=1)) < 1e-10:
            continue

        eps = 1e-6
        if eps < best_t < (1.0 - eps):
            out = np.insert(out, best_i + 1, best_q, axis=0)
    return out




def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """単純多角形の面積重心。面積が極小なら頂点平均を返す。"""
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    x = p[:, 0]
    y = p[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cross = x * y2 - x2 * y
    A2 = float(np.sum(cross))
    if abs(A2) < 1e-12:
        return np.array([float(np.mean(x)), float(np.mean(y))], dtype=np.float64)
    cx = float(np.sum((x + x2) * cross) / (3.0 * A2))
    cy = float(np.sum((y + y2) * cross) / (3.0 * A2))
    return np.array([cx, cy], dtype=np.float64)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def ray_segment_intersection(
    C: np.ndarray,
    d: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    eps: float = 1e-12,
) -> Optional[Tuple[float, float, np.ndarray]]:
    """
    半直線 C + t d (t>=0) と線分 AB の交点。
    戻り値: (t, u, P) ただし P = A + u(B-A), u in [0,1]
    """
    r = d
    s = B - A
    denom = _cross2(r, s)
    if abs(denom) < eps:
        return None
    AC = A - C
    t = _cross2(AC, s) / denom
    u = _cross2(AC, r) / denom
    if t < 0.0:
        return None
    if u < -1e-9 or u > 1.0 + 1e-9:
        return None
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    P = A + u * s
    return float(t), float(u), P


def ray_polygon_first_intersection(
    poly: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
) -> Optional[Tuple[int, float, np.ndarray, float]]:
    """
    半直線と多角形境界の交点を全探索し、距離が最も遠い（t最大）ものを返す。
    戻り値: (seg_i, u, P, t)
      - seg_i: 交差した辺の開始頂点 index i（辺 i -> (i+1)%n）
      - u: 辺上のパラメータ
      - P: 交点
      - t: 半直線パラメータ
    """
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    best = None
    best_t = -float("inf")
    for i in range(n):
        A = p[i]
        B = p[(i + 1) % n]
        hit = ray_segment_intersection(C, d, A, B)
        if hit is None:
            continue
        t, u, P = hit
        if t > best_t:
            best_t = t
            best = (i, u, P, t)
    return best


def rect_point_to_perimeter_s(u: float, v: float, W: float, H: float) -> float:
    """矩形境界上の点 (u,v) を周長パラメータ s∈[0,2(W+H)) に変換（上→右→下→左）。"""
    eps = 1e-6
    if abs(v - 0.0) <= eps:
        return float(np.clip(u, 0.0, W))
    if abs(u - W) <= eps:
        return float(W + np.clip(v, 0.0, H))
    if abs(v - H) <= eps:
        return float(W + H + (W - np.clip(u, 0.0, W)))
    # left
    return float(2.0 * W + H + (H - np.clip(v, 0.0, H)))


def rect_perimeter_s_to_uv(s: float, W: float, H: float) -> np.ndarray:
    """周長パラメータ s から矩形境界上の (u,v) を返す。"""
    P = 2.0 * (W + H)
    s = float(s % P)
    if s < W:
        return np.array([s, 0.0], dtype=np.float64)
    s -= W
    if s < H:
        return np.array([W, s], dtype=np.float64)
    s -= H
    if s < W:
        return np.array([W - s, H], dtype=np.float64)
    s -= W
    return np.array([0.0, H - s], dtype=np.float64)


def ray_rectangle_first_intersection(
    C: np.ndarray,
    d: np.ndarray,
    W: float,
    H: float,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    半直線と矩形境界（(0,0)-(W,H)）の交点を返す（距離が近い方）。
    戻り値: (P, s) ただし s は周長パラメータ
    """
    # 4辺を線分として扱う
    edges = [
        (np.array([0.0, 0.0]), np.array([W, 0.0])),
        (np.array([W, 0.0]), np.array([W, H])),
        (np.array([W, H]), np.array([0.0, H])),
        (np.array([0.0, H]), np.array([0.0, 0.0])),
    ]
    best_t = float("inf")
    best_P = None
    for A, B in edges:
        hit = ray_segment_intersection(C, d, A, B)
        if hit is None:
            continue
        t, u, P = hit
        if t < best_t:
            best_t = t
            best_P = P
    if best_P is None:
        return None
    s = rect_point_to_perimeter_s(float(best_P[0]), float(best_P[1]), W, H)
    return best_P, s


def insert_ray_intersections(poly: np.ndarray, center: np.ndarray, step_deg: float) -> np.ndarray:
    """
    多角形中心 center から一定角度刻みの半直線を引き、各半直線について
    多角形境界との最遠交点（t最大）を求め、交点が辺内部なら頂点として挿入する。
    - 複数交点がある場合は距離が遠い方（t最大）を採用。
    """
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    if n < 3:
        return p

    step = float(step_deg)
    if step <= 0.0:
        return p

    # 角度は screen 座標系（x右, y下）で atan2 を想定。0deg は +x 方向。
    angles = np.arange(0.0, 360.0, step, dtype=np.float64)
    hits_by_seg: Dict[int, List[Tuple[float, np.ndarray]]] = {}

    for deg in angles:
        th = math.radians(float(deg))
        d = np.array([math.cos(th), math.sin(th)], dtype=np.float64)
        hit = ray_polygon_first_intersection(p, center, d)
        if hit is None:
            continue
        seg_i, u, P, _t = hit
        eps = 1e-6
        if not (eps < u < (1.0 - eps)):
            continue  # 端点近傍は既存頂点とみなす
        hits_by_seg.setdefault(int(seg_i), []).append((float(u), P))

    if not hits_by_seg:
        return p

    # 各辺内で u 昇順に挿入
    out: List[np.ndarray] = []
    for i in range(n):
        out.append(p[i])
        pts = hits_by_seg.get(i, [])
        if pts:
            pts.sort(key=lambda t: t[0])
            for _u, P in pts:
                # 直前と近すぎる場合はスキップ
                if np.linalg.norm(P - out[-1]) < 1e-6:
                    continue
                out.append(P)

    # 閉路の最後→先頭辺に挿入がある場合、末尾に追加済み（i=n-1 で処理）
    return np.asarray(out, dtype=np.float64)


def uv_boundary_rays(
    boundary_poly: np.ndarray,
    Ws: int,
    Hs: int,
    ray_step_deg: float,
    poly_center: Optional[np.ndarray] = None,
    return_anchors: bool = False,
):
    """
    多角形中心から一定角度刻みの半直線を引き、
      - 半直線と多角形境界の最遠交点（複数なら距離が遠い方）
      - 半直線と元画像矩形境界の交点（矩形中心から同角度、矩形境界との交点は通常1つ）
    を対応させる境界条件を作る。

    return_anchors=True のときは、
      (uv_boundary, poly_anchor_points, rect_anchor_points, labels)
    を返す。
    """
    bp = np.asarray(boundary_poly, dtype=np.float64)
    n = len(bp)
    if n < 3:
        uv_b0 = np.zeros((n, 2), dtype=np.float64)
        if return_anchors:
            return uv_b0, np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64), []
        return uv_b0

    step = float(ray_step_deg)
    if step <= 0.0:
        step = 30.0

    # poly 中心（指定がなければ境界から）
    Cpoly = np.asarray(poly_center, dtype=np.float64) if poly_center is not None else polygon_centroid(bp)

    # rectangle（元画像）中心
    W = float(Ws - 1)
    H = float(Hs - 1)
    Crect = np.array([W * 0.5, H * 0.5], dtype=np.float64)
    Pper = 2.0 * (W + H)

    # 境界の累積長
    loop = np.vstack([bp, bp[:1]])
    s_cum = cumulative_arclength(loop)  # len n+1
    L = float(s_cum[-1])
    if L < 1e-12:
        uv_b0 = np.zeros((n, 2), dtype=np.float64)
        if return_anchors:
            return uv_b0, np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64), []
        return uv_b0

    angles = np.arange(0.0, 360.0, step, dtype=np.float64)

    # anchors: (s_poly, s_rect) + 可視化用点
    anchors: List[Tuple[float, float]] = []
    poly_pts: List[np.ndarray] = []
    rect_pts: List[np.ndarray] = []
    labels: List[str] = []

    for deg in angles:
        th = math.radians(float(deg))
        d = np.array([math.cos(th), math.sin(th)], dtype=np.float64)

        # polygon intersection on boundary_poly segments（閉路）: 最遠（t最大）
        best_t = -float("inf")
        best_s_poly = None
        best_Ppoly = None

        for i in range(n):
            A = bp[i]
            B = bp[(i + 1) % n]
            hit = ray_segment_intersection(Cpoly, d, A, B)
            if hit is None:
                continue
            t, u, P = hit
            if t > best_t:
                best_t = t
                seg_len = float(np.linalg.norm(B - A))
                best_s_poly = float(s_cum[i] + u * seg_len)
                best_Ppoly = P

        if best_s_poly is None or best_Ppoly is None:
            continue

        # rectangle intersection (from rectangle center)
        rh = ray_rectangle_first_intersection(Crect, d, W, H)
        if rh is None:
            continue
        Pr, s_rect = rh

        anchors.append((best_s_poly, float(s_rect)))
        poly_pts.append(best_Ppoly.astype(np.float64))
        rect_pts.append(Pr.astype(np.float64))
        labels.append(f"{int(round(deg))}°")

    if len(anchors) < 2:
        # 充分なアンカーが得られない場合はフォールバック
        uv_b = uv_boundary_arclength(bp, Ws=Ws, Hs=Hs)
        if return_anchors:
            return uv_b, np.asarray(poly_pts, dtype=np.float64), np.asarray(rect_pts, dtype=np.float64), labels
        return uv_b

    # s_poly でソートし、s_rect を単調増加にアンラップ
    order = np.argsort([a[0] for a in anchors])
    anchors_sorted = [anchors[i] for i in order]
    poly_pts_sorted = [poly_pts[i] for i in order]
    rect_pts_sorted = [rect_pts[i] for i in order]
    labels_sorted = [labels[i] for i in order]

    sp = np.array([a[0] for a in anchors_sorted], dtype=np.float64)
    sr = np.array([a[1] for a in anchors_sorted], dtype=np.float64)

    sr_un = np.empty_like(sr)
    sr_un[0] = sr[0]
    for i in range(1, len(sr)):
        val = sr[i]
        while val < sr_un[i - 1] - 1e-9:
            val += Pper
        sr_un[i] = val

    # 拡張配列（wrap 部）
    sp_ext = np.concatenate([sp, sp[:1] + L])
    sr_ext = np.concatenate([sr_un, sr_un[:1] + Pper])

    # 各境界頂点の s
    s_v = s_cum[:-1].copy()

    uv_b = np.zeros((n, 2), dtype=np.float64)
    for idx in range(n):
        s = float(s_v[idx])
        s2 = s + L if s < sp_ext[0] else s

        # 区間探索
        j = int(np.searchsorted(sp_ext, s2, side="right") - 1)
        if j < 0:
            j = 0
        if j >= len(sp_ext) - 1:
            j = len(sp_ext) - 2

        s0, s1 = float(sp_ext[j]), float(sp_ext[j + 1])
        r0, r1 = float(sr_ext[j]), float(sr_ext[j + 1])
        if abs(s1 - s0) < 1e-12:
            r = r0
        else:
            a = (s2 - s0) / (s1 - s0)
            r = r0 * (1.0 - a) + r1 * a

        uv = rect_perimeter_s_to_uv(r, W, H)
        uv_b[idx] = uv

    if not return_anchors:
        return uv_b

    return uv_b, np.asarray(poly_pts_sorted, dtype=np.float64), np.asarray(rect_pts_sorted, dtype=np.float64), labels_sorted


def rotate_boundary_order_to_nearest_point(boundary_order: np.ndarray, boundary_poly: np.ndarray, p: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    boundary_poly のうち点pに最も近い頂点が先頭になるように boundary_order を回転する。
    """
    q = np.array(p, dtype=np.float64)
    k = int(np.argmin(np.sum((boundary_poly - q) ** 2, axis=1)))
    if k == 0:
        return boundary_order, boundary_poly
    bo = np.concatenate([boundary_order[k:], boundary_order[:k]])
    bp = np.concatenate([boundary_poly[k:], boundary_poly[:k]], axis=0)
    return bo, bp


def uv_boundary_arclength(boundary_poly: np.ndarray, Ws: int, Hs: int) -> np.ndarray:
    """
    境界頂点列（閉じていない）を弧長で 0..1 に正規化して矩形周へマップ。
    """
    loop = np.vstack([boundary_poly, boundary_poly[:1]])
    s = cumulative_arclength(loop)
    L = float(s[-1])
    if L < 1e-12:
        raise RuntimeError("Boundary length is zero.")
    t = (s[:-1] / L)
    return map_t_to_rectangle_perimeter(t, W=float(Ws - 1), H=float(Hs - 1))


def uv_boundary_anchors8_nearest(
    boundary_poly: np.ndarray,
    canvas_size: Tuple[int, int],
    Ws: int,
    Hs: int,
    return_anchors: bool = False,
):
    """
    4隅 + 辺中点（計8点）を「境界上の最近点」に対応させるように、
    境界の弧長→矩形周の対応を “アンカー付きの区分線形” で作る。

    return_anchors=True のときは、
      (uv_boundary, poly_anchor_points, rect_anchor_points, labels)
    を返す。
    """
    Wo, Ho = canvas_size
    # アンカー点（キャンバス座標）: TL, TM, TR, RM, BR, BM, BL, LM
    anchors_xy = [
        (0.0, 0.0),
        ((Wo - 1) * 0.5, 0.0),
        (float(Wo - 1), 0.0),
        (float(Wo - 1), (Ho - 1) * 0.5),
        (float(Wo - 1), float(Ho - 1)),
        ((Wo - 1) * 0.5, float(Ho - 1)),
        (0.0, float(Ho - 1)),
        (0.0, (Ho - 1) * 0.5),
    ]
    labels = ["TL", "TM", "TR", "RM", "BR", "BM", "BL", "LM"]

    # 矩形周上の距離 s_rect（UV側）
    W = float(Ws - 1)
    H = float(Hs - 1)
    P = 2.0 * (W + H)
    rect_s = np.array([
        0.0,
        0.5 * W,
        1.0 * W,
        1.0 * W + 0.5 * H,
        1.0 * W + 1.0 * H,
        1.0 * W + 1.0 * H + 0.5 * W,
        2.0 * W + 1.0 * H,
        2.0 * W + 1.0 * H + 0.5 * H,
    ], dtype=np.float64)

    # 対応させる矩形側の点（UV座標）
    rect_anchor_uv = np.array([
        [0.0, 0.0],
        [0.5 * W, 0.0],
        [W, 0.0],
        [W, 0.5 * H],
        [W, H],
        [0.5 * W, H],
        [0.0, H],
        [0.0, 0.5 * H],
    ], dtype=np.float64)

    # 境界の累積長（boundary_poly は閉じていない列）
    loop = np.vstack([boundary_poly, boundary_poly[:1]])
    s = cumulative_arclength(loop)
    L = float(s[-1])
    if L < 1e-12:
        raise RuntimeError("Boundary length is zero.")
    s_vertices = s[:-1]  # 各頂点の弧長位置（0..L）

    # 各アンカーに対して「境界上最近点」を求め、その点に最も近い境界頂点をアンカー頂点とする（重複は避ける）
    used = set()
    anchor_idx: List[int] = []
    anchor_s: List[float] = []

    for ax, ay in anchors_xy:
        p = np.array([ax, ay], dtype=np.float64)
        q, _ = closest_point_on_polyline(p, boundary_poly)

        # 候補を距離順に並べて未使用を選ぶ
        d2 = np.sum((boundary_poly - q) ** 2, axis=1)
        cand = np.argsort(d2)
        picked = None
        for ci in cand:
            ci = int(ci)
            if ci not in used:
                picked = ci
                break
        if picked is None:
            picked = int(cand[0])

        used.add(picked)
        anchor_idx.append(picked)
        anchor_s.append(float(s_vertices[picked]))

    anchor_s = np.array(anchor_s, dtype=np.float64)

    # 先頭（TL）を 0 とみなし、相対位置へ
    s0 = anchor_s[0]
    rel = (anchor_s - s0) % L

    # アンカー順に単調増加になるよう軽く整形（同一点/逆転でゼロ割にならないよう）
    eps = 1e-5 * L
    rel_sorted = rel.copy()
    for k in range(1, len(rel_sorted)):
        if rel_sorted[k] <= rel_sorted[k - 1] + eps:
            rel_sorted[k] = rel_sorted[k - 1] + eps

    # 最後の終点を追加（TLに戻る）
    rel_ext = np.concatenate([rel_sorted, [L]])
    rect_ext = np.concatenate([rect_s, [P]])

    # 各境界頂点の s（既に0..Lで、先頭を基準に回転済みなので rel = s_vertices - s_vertices[0]）
    s_rel_vertices = (s_vertices - s_vertices[0]) % L

    # 区分線形写像：s_rel -> rect_s
    rect_s_v = np.zeros_like(s_rel_vertices)
    for i, sv in enumerate(s_rel_vertices):
        k = int(np.searchsorted(rel_ext, sv, side="right") - 1)
        k = max(0, min(k, len(rel_ext) - 2))
        s_a = rel_ext[k]
        s_b = rel_ext[k + 1]
        r_a = rect_ext[k]
        r_b = rect_ext[k + 1]
        if abs(s_b - s_a) < 1e-12:
            rect_s_v[i] = r_a
        else:
            alpha = (sv - s_a) / (s_b - s_a)
            rect_s_v[i] = r_a + alpha * (r_b - r_a)

    t = rect_s_v / P
    uv_b = map_t_to_rectangle_perimeter(t, W=float(Ws - 1), H=float(Hs - 1))

    if not return_anchors:
        return uv_b

    poly_anchor_pts = boundary_poly[np.array(anchor_idx, dtype=np.int32)]
    return uv_b, poly_anchor_pts, rect_anchor_uv, labels


def uv_boundary_anchors4_nearest(
    boundary_poly: np.ndarray,
    canvas_size: Tuple[int, int],
    Ws: int,
    Hs: int,
    return_anchors: bool = False,
):
    """
    4隅（TL, TR, BR, BL）を「境界上の最近点」に対応させるように、
    境界の弧長→矩形周の対応を “アンカー付きの区分線形” で作る。

    return_anchors=True のときは、
      (uv_boundary, poly_anchor_points, rect_anchor_points, labels)
    を返す。
    """
    Wo, Ho = canvas_size

    # アンカー点（キャンバス座標）: TL, TR, BR, BL
    anchors_xy = [
        (0.0, 0.0),
        (float(Wo - 1), 0.0),
        (float(Wo - 1), float(Ho - 1)),
        (0.0, float(Ho - 1)),
    ]
    labels = ["TL", "TR", "BR", "BL"]

    # 矩形周上の距離 s_rect（UV側）: TL=0, TR=W, BR=W+H, BL=2W+H
    W = float(Ws - 1)
    H = float(Hs - 1)
    P = 2.0 * (W + H)
    rect_s = np.array([
        0.0,
        1.0 * W,
        1.0 * W + 1.0 * H,
        2.0 * W + 1.0 * H,
    ], dtype=np.float64)

    rect_anchor_uv = np.array([
        [0.0, 0.0],
        [W, 0.0],
        [W, H],
        [0.0, H],
    ], dtype=np.float64)

    # 境界の累積長（boundary_poly は閉じていない列）
    loop = np.vstack([boundary_poly, boundary_poly[:1]])
    s = cumulative_arclength(loop)
    L = float(s[-1])
    if L < 1e-12:
        raise RuntimeError("Boundary length is zero.")
    s_vertices = s[:-1]

    # 各アンカーに対して「境界上最近点」を求め、その点に最も近い境界頂点をアンカー頂点とする（重複は避ける）
    used = set()
    anchor_idx: List[int] = []
    anchor_s: List[float] = []
    for ax, ay in anchors_xy:
        p = np.array([ax, ay], dtype=np.float64)
        q, _ = closest_point_on_polyline(p, boundary_poly)

        d2 = np.sum((boundary_poly - q) ** 2, axis=1)
        cand = np.argsort(d2)
        picked = None
        for ci in cand:
            ci = int(ci)
            if ci not in used:
                picked = ci
                break
        if picked is None:
            picked = int(cand[0])
        used.add(picked)
        anchor_idx.append(picked)
        anchor_s.append(float(s_vertices[picked]))

    anchor_s = np.array(anchor_s, dtype=np.float64)

    # TL を 0 とみなし、相対位置へ
    s0 = anchor_s[0]
    rel = (anchor_s - s0) % L

    # アンカー順に単調増加になるよう軽く整形
    eps = 1e-5 * L
    rel_sorted = rel.copy()
    for k in range(1, len(rel_sorted)):
        if rel_sorted[k] <= rel_sorted[k - 1] + eps:
            rel_sorted[k] = rel_sorted[k - 1] + eps

    # 終点を追加（TLに戻る）
    rel_ext = np.concatenate([rel_sorted, [L]])
    rect_ext = np.concatenate([rect_s, [P]])

    # 各境界頂点の相対弧長
    s_rel_vertices = (s_vertices - s_vertices[0]) % L

    # 区分線形写像：s_rel -> rect_s
    rect_s_v = np.zeros_like(s_rel_vertices)
    for i, sv in enumerate(s_rel_vertices):
        k = int(np.searchsorted(rel_ext, sv, side="right") - 1)
        k = max(0, min(k, len(rel_ext) - 2))
        s_a = rel_ext[k]
        s_b = rel_ext[k + 1]
        r_a = rect_ext[k]
        r_b = rect_ext[k + 1]
        if abs(s_b - s_a) < 1e-12:
            rect_s_v[i] = r_a
        else:
            alpha = (sv - s_a) / (s_b - s_a)
            rect_s_v[i] = r_a + alpha * (r_b - r_a)

    t = rect_s_v / P
    uv_b = map_t_to_rectangle_perimeter(t, W=float(Ws - 1), H=float(Hs - 1))

    if not return_anchors:
        return uv_b

    poly_anchor_pts = boundary_poly[np.array(anchor_idx, dtype=np.int32)]
    return uv_b, poly_anchor_pts, rect_anchor_uv, labels


def compute_boundary_uv(
    boundary_order: np.ndarray,
    boundary_poly: np.ndarray,
    canvas_size: Tuple[int, int],
    Ws: int,
    Hs: int,
    mode: str,
    ray_step_deg: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    mode:
      - "origin":    原点(0,0)に最近い境界点を起点（t=0）にした弧長→矩形周
      - "anchors4":  4隅（TL,TR,BR,BL）を「最近点」に対応させるアンカー付き対応
      - "anchors8":  4隅+辺中点（計8点）を「最近点」に対応させるアンカー付き対応
      - "rays":      多角形中心から一定角度刻みの半直線で得た交点どうしを対応

    戻り値:
      (boundary_order_rot, boundary_poly_rot, uv_boundary,
       corresp_poly_points, corresp_rect_points, corresp_labels)
    """
    mode = (mode or "origin").lower().strip()
    if mode not in ("origin", "anchors4", "anchors8", "rays"):
        mode = "origin"

    cor_poly = np.zeros((0, 2), dtype=np.float64)
    cor_rect = np.zeros((0, 2), dtype=np.float64)
    cor_labels: List[str] = []

    if mode in ("origin", "anchors4", "anchors8"):
        boundary_order, boundary_poly = rotate_boundary_order_to_nearest_point(boundary_order, boundary_poly, (0.0, 0.0))

    if mode == "anchors8":
        uv_b, cor_poly, cor_rect, cor_labels = uv_boundary_anchors8_nearest(
            boundary_poly, canvas_size=canvas_size, Ws=Ws, Hs=Hs, return_anchors=True
        )
    elif mode == "anchors4":
        uv_b, cor_poly, cor_rect, cor_labels = uv_boundary_anchors4_nearest(
            boundary_poly, canvas_size=canvas_size, Ws=Ws, Hs=Hs, return_anchors=True
        )
    elif mode == "rays":
        uv_b, cor_poly, cor_rect, cor_labels = uv_boundary_rays(
            boundary_poly, Ws=Ws, Hs=Hs, ray_step_deg=ray_step_deg, return_anchors=True
        )
    else:
        uv_b = uv_boundary_arclength(boundary_poly, Ws=Ws, Hs=Hs)
        # origin は boundary_poly[0] が (0,0) 最近点になるよう事前挿入＋回転している前提
        W = float(Ws - 1)
        H = float(Hs - 1)
        cor_poly = boundary_poly[:1].copy()
        cor_rect = np.array([[0.0, 0.0]], dtype=np.float64)
        cor_labels = ["O"]

    # "origin" モードは arclength を使い、(0,0) に最近い点が先頭に来るようにしている
    if mode == "origin":
        uv_b = uv_boundary_arclength(boundary_poly, Ws=Ws, Hs=Hs)
        cor_poly = boundary_poly[:1].copy()
        cor_rect = np.array([[0.0, 0.0]], dtype=np.float64)
        cor_labels = ["O"]

    return boundary_order, boundary_poly, uv_b, cor_poly, cor_rect, cor_labels


def map_t_to_rectangle_perimeter(t: np.ndarray, W: float, H: float) -> np.ndarray:
    """
    t in [0,1) を矩形周（反時計回り）に写像して (u,v) を返す。
    矩形は (0,0)-(W,H)（ピクセル座標系）。
    """
    P = 2.0 * (W + H)
    s = (t % 1.0) * P
    uv = np.zeros((len(t), 2), dtype=np.float64)

    m = s < W
    uv[m, 0] = s[m]
    uv[m, 1] = 0.0

    m2 = (s >= W) & (s < W + H)
    uv[m2, 0] = W
    uv[m2, 1] = s[m2] - W

    m3 = (s >= W + H) & (s < 2.0 * W + H)
    uv[m3, 0] = W - (s[m3] - (W + H))
    uv[m3, 1] = H

    m4 = s >= (2.0 * W + H)
    uv[m4, 0] = 0.0
    uv[m4, 1] = H - (s[m4] - (2.0 * W + H))
    return uv

def cotangent(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(a[0]*b[0] + a[1]*b[1])
    cross = float(a[0]*b[1] - a[1]*b[0])
    eps = 1e-12
    if abs(cross) < eps:
        return 0.0
    return dot / cross

def build_cotangent_weights(V: np.ndarray, F: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    各エッジ (i,j) に対して (cot α + cot β)/2 を構築（境界は片側のみ）。
    """
    w: Dict[Tuple[int, int], float] = {}
    for (i, j, k) in F.astype(int):
        pi, pj, pk = V[i], V[j], V[k]
        cot_i = cotangent(pj - pi, pk - pi)
        cot_j = cotangent(pk - pj, pi - pj)
        cot_k = cotangent(pi - pk, pj - pk)

        for (a, b, c) in [(j, k, cot_i), (k, i, cot_j), (i, j, cot_k)]:
            aa, bb = (a, b) if a < b else (b, a)
            w[(aa, bb)] = w.get((aa, bb), 0.0) + c

    for key in list(w.keys()):
        w[key] *= 0.5
    return w

def build_laplacian_system(
    n: int,
    weights: Dict[Tuple[int, int], float],
    boundary: np.ndarray,
    uv_boundary: np.ndarray,
):
    """
    内部頂点のみ未知数として A x = b を作る。
    Σ w_ij (U_i - U_j)=0 を離散化し、境界項は右辺へ移す。
    """
    boundary_set = set(int(i) for i in boundary)
    interior_idx = np.array([i for i in range(n) if i not in boundary_set], dtype=np.int32)
    k = len(interior_idx)

    Ufix = np.zeros((n, 2), dtype=np.float64)
    for idx, uv in zip(boundary, uv_boundary):
        Ufix[int(idx), :] = uv

    neigh: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for (i, j), wij in weights.items():
        i = int(i); j = int(j)
        if wij == 0.0:
            continue
        neigh[i].append((j, wij))
        neigh[j].append((i, wij))

    row_of = {int(v): r for r, v in enumerate(interior_idx)}

    rows, cols, data = [], [], []
    bu = np.zeros(k, dtype=np.float64)
    bv = np.zeros(k, dtype=np.float64)

    for r, vi in enumerate(interior_idx):
        vi = int(vi)
        diag = 0.0
        for vj, wij in neigh[vi]:
            diag += wij
            if vj in boundary_set:
                bu[r] += wij * Ufix[vj, 0]
                bv[r] += wij * Ufix[vj, 1]
            else:
                c = row_of[int(vj)]
                rows.append(r); cols.append(c); data.append(-wij)

        rows.append(r); cols.append(r); data.append(diag)

    A = coo_matrix((data, (rows, cols)), shape=(k, k)).tocsr()
    return A, bu, bv, interior_idx

def bilinear_sample(src: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    Hs, Ws, C = src.shape
    u = np.clip(u, 0.0, Ws - 1.000001)
    v = np.clip(v, 0.0, Hs - 1.000001)

    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, Ws - 1)
    y1 = np.clip(y0 + 1, 0, Hs - 1)

    du = (u - x0)[..., None]
    dv = (v - y0)[..., None]

    Ia = src[y0, x0]
    Ib = src[y0, x1]
    Ic = src[y1, x0]
    Id = src[y1, x1]

    Iab = Ia * (1 - du) + Ib * du
    Icd = Ic * (1 - du) + Id * du
    return Iab * (1 - dv) + Icd * dv

def rasterize_triangle_uv(
    out: np.ndarray,
    src: np.ndarray,
    A: np.ndarray, B: np.ndarray, C: np.ndarray,   # XY
    a: np.ndarray, b: np.ndarray, c: np.ndarray,   # UV
):
    Ho, Wo, _ = out.shape
    xmin = int(max(0, math.floor(min(A[0], B[0], C[0]))))
    xmax = int(min(Wo - 1, math.ceil(max(A[0], B[0], C[0]))))
    ymin = int(max(0, math.floor(min(A[1], B[1], C[1]))))
    ymax = int(min(Ho - 1, math.ceil(max(A[1], B[1], C[1]))))
    if xmax < xmin or ymax < ymin:
        return

    xs = np.arange(xmin, xmax + 1, dtype=np.float64)
    ys = np.arange(ymin, ymax + 1, dtype=np.float64)
    X, Y = np.meshgrid(xs + 0.5, ys + 0.5)

    v0 = B - A
    v1 = C - A
    den = v0[0]*v1[1] - v0[1]*v1[0]
    if abs(den) < 1e-12:
        return

    v2x = X - A[0]
    v2y = Y - A[1]

    lB = (v2x * v1[1] - v2y * v1[0]) / den
    lC = (v0[0] * v2y - v0[1] * v2x) / den
    lA = 1.0 - lB - lC

    eps = -1e-6
    mask = (lA >= eps) & (lB >= eps) & (lC >= eps)
    if not np.any(mask):
        return

    U = lA * a[0] + lB * b[0] + lC * c[0]
    V = lA * a[1] + lB * b[1] + lC * c[1]

    sampled = bilinear_sample(src, U, V)

    yy, xx = np.where(mask)
    out[ymin:ymax+1, xmin:xmax+1][yy, xx] = sampled[yy, xx]


@dataclass
class HarmonicWarpResult:
    out_image: Image.Image
    mesh_vertices_xy: np.ndarray
    mesh_vertices_uv: np.ndarray
    triangles: np.ndarray
    segments: np.ndarray
    boundary_order: np.ndarray

    # 境界条件で用いた「対応点」可視化用（poly側は出力キャンバス座標、rect側は入力画像UV座標）
    correspondence_poly_points: np.ndarray  # (k,2)
    correspondence_rect_points: np.ndarray  # (k,2)
    correspondence_labels: List[str]


def warp_image_to_polygon_harmonic(
    src_img: Image.Image,
    polygon_xy: np.ndarray,
    canvas_size: Tuple[int, int],
    max_area: Optional[float] = None,
    min_angle: float = 28.0,
    boundary_mode: str = "origin",
    ray_step_deg: float = 30.0,
) -> HarmonicWarpResult:
    """
    GUI用：出力キャンバスサイズを固定し、polygon_xy はそのキャンバス座標系で与える。
    """
    # 境界条件モードに応じて、対応させたい「境界上の最近点」を頂点として挿入しておく
    mode = (boundary_mode or "origin").lower().strip()
    poly_raw = np.asarray(polygon_xy, dtype=np.float64)

    if mode == "origin":
        # 原点(0,0) と多角形境界上の最近点を対応させたい
        poly_raw = insert_closest_boundary_points(poly_raw, [(0.0, 0.0)])
    elif mode == "anchors4":
        # 4隅（TL,TR,BR,BL）と、多角形境界上の最近点を対応させたい
        Wo_tmp, Ho_tmp = canvas_size
        targets = [
            (0.0, 0.0),
            (float(Wo_tmp - 1), 0.0),
            (float(Wo_tmp - 1), float(Ho_tmp - 1)),
            (0.0, float(Ho_tmp - 1)),
        ]
        poly_raw = insert_closest_boundary_points(poly_raw, targets)
    elif mode == "anchors8":
        # 4隅+辺中点（計8点）と、多角形境界上の最近点を対応させたい
        Wo_tmp, Ho_tmp = canvas_size
        targets = [
            (0.0, 0.0),
            ((Wo_tmp - 1) * 0.5, 0.0),
            (float(Wo_tmp - 1), 0.0),
            (float(Wo_tmp - 1), (Ho_tmp - 1) * 0.5),
            (float(Wo_tmp - 1), float(Ho_tmp - 1)),
            ((Wo_tmp - 1) * 0.5, float(Ho_tmp - 1)),
            (0.0, float(Ho_tmp - 1)),
            (0.0, (Ho_tmp - 1) * 0.5),
        ]
        poly_raw = insert_closest_boundary_points(poly_raw, targets)

    elif mode == "rays":
        # 多角形中心から一定角度刻みの半直線で得た交点を境界条件として固定したい
        ctmp = polygon_centroid(poly_raw)
        poly_raw = insert_ray_intersections(poly_raw, center=ctmp, step_deg=ray_step_deg)

    poly = ensure_ccw(poly_raw)

    Wo, Ho = canvas_size
    if Wo <= 0 or Ho <= 0:
        raise ValueError("Invalid canvas_size.")

    # max_area 自動推定
    if max_area is None:
        bbox_area = float(Wo * Ho)
        target_tris = 5000.0
        max_area = max(bbox_area / target_tris, 1.0)

    # 制約付き三角形分割（PSLG）
    A = build_pslg(poly)
    opts = f"pq{min_angle}a{max_area}zQ"
    T = tr.triangulate(A, opts)

    Vxy = T["vertices"].astype(np.float64)
    F = T["triangles"].astype(np.int32)
    seg = T.get("segments", None)
    if seg is None:
        raise RuntimeError("triangle output has no segments.")
    seg = seg.astype(np.int32)

    # 境界順序
    boundary_order = np.array(segments_boundary_order(Vxy, seg), dtype=np.int32)
    boundary_poly = Vxy[boundary_order]
    if polygon_signed_area(boundary_poly) < 0:
        boundary_order = boundary_order[::-1].copy()
        boundary_poly = boundary_poly[::-1].copy()

    # 境界条件の起点(t=0)を「原点(0,0)に最も近い境界上の点」に合わせる。
    # triangle が境界上にSteiner点を追加している場合も、境界頂点の中で最近の点を採用する。
    d2 = np.sum(boundary_poly * boundary_poly, axis=1)  # (x^2 + y^2)
    start_idx = int(np.argmin(d2))
    if start_idx != 0:
        boundary_order = np.roll(boundary_order, -start_idx)
        boundary_poly = Vxy[boundary_order]

    # 入力画像（RGBA）
    src_img = src_img.convert("RGBA")
    src = np.asarray(src_img).astype(np.float64) / 255.0
    Hs, Ws, Cs = src.shape

    # 境界UV固定：モードに応じて生成（必要なら起点回転も行う）
    boundary_order, boundary_poly, uv_b, cor_poly, cor_rect, cor_labels = compute_boundary_uv(
        boundary_order=boundary_order,
        boundary_poly=boundary_poly,
        canvas_size=canvas_size,
        Ws=Ws,
        Hs=Hs,
        mode=mode,
        ray_step_deg=ray_step_deg,
    )

    # U初期化（境界固定）
    U = np.zeros((len(Vxy), 2), dtype=np.float64)
    U[boundary_order] = uv_b

    # cotangent Laplacian → 内部UVを解く
    weights = build_cotangent_weights(Vxy, F)
    A_sys, bu, bv, interior_idx = build_laplacian_system(
        n=len(Vxy),
        weights=weights,
        boundary=boundary_order,
        uv_boundary=uv_b,
    )
    if A_sys.shape[0] > 0:
        uu = spsolve(A_sys, bu)
        vv = spsolve(A_sys, bv)
        U[interior_idx, 0] = uu
        U[interior_idx, 1] = vv

    # ラスタライズ
    out = np.zeros((Ho, Wo, Cs), dtype=np.float64)
    for (i, j, k) in F:
        Axy, Bxy, Cxy = Vxy[i], Vxy[j], Vxy[k]
        auv, buv, cuv = U[i], U[j], U[k]
        rasterize_triangle_uv(out, src, Axy, Bxy, Cxy, auv, buv, cuv)

    out = np.clip(out, 0.0, 1.0)
    out_img = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGBA")

    return HarmonicWarpResult(
        out_image=out_img,
        mesh_vertices_xy=Vxy,
        mesh_vertices_uv=U,
        triangles=F,
        segments=seg,
        boundary_order=boundary_order,
        correspondence_poly_points=cor_poly,
        correspondence_rect_points=cor_rect,
        correspondence_labels=cor_labels,
    )


def render_mesh_overlay_image(
    vertices_xy: np.ndarray,
    triangles: np.ndarray,
    segments: np.ndarray,
    boundary_order: np.ndarray,
    canvas_size: Tuple[int, int],
    base_img: Optional[Image.Image] = None,  # ワープ結果など（出力座標系に合うもの）
    base_alpha: float = 0.35,
    edge_width: int = 1,
    boundary_width: int = 3,
    vertex_radius: int = 2,
) -> Image.Image:
    W, H = canvas_size
    V = vertices_xy.astype(np.float64)

    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    if base_img is not None:
        bg = base_img.convert("RGBA")
        if bg.size != (W, H):
            bg = bg.resize((W, H), Image.BILINEAR)
        a = int(max(0, min(255, round(base_alpha * 255))))
        alpha = bg.getchannel("A")
        alpha = alpha.point(lambda p: (p * a) // 255)
        bg.putalpha(alpha)
        canvas = Image.alpha_composite(canvas, bg)

    draw = ImageDraw.Draw(canvas)

    # 全エッジ（細線）
    edges = set()
    for i, j, k in triangles.astype(np.int32):
        a, b, c = int(i), int(j), int(k)
        e0 = (a, b) if a < b else (b, a)
        e1 = (b, c) if b < c else (c, b)
        e2 = (c, a) if c < a else (a, c)
        edges.add(e0); edges.add(e1); edges.add(e2)

    for a, b in edges:
        x0, y0 = V[a]
        x1, y1 = V[b]
        draw.line((x0, y0, x1, y1), fill=(0, 0, 0, 255), width=edge_width)

    # 境界（segments：太線）
    for a, b in segments.astype(np.int32):
        a = int(a); b = int(b)
        x0, y0 = V[a]
        x1, y1 = V[b]
        draw.line((x0, y0, x1, y1), fill=(0, 0, 0, 255), width=boundary_width)

    # 境界頂点（点）
    r = vertex_radius
    for idx in boundary_order.astype(np.int32):
        x, y = V[int(idx)]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0, 255))

    return canvas


# ============================================================
#  GUI
# ============================================================

class HarmonicWarpGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Harmonic Warp (Polygon → Image)")

        # 状態
        self.src_img: Optional[Image.Image] = None
        self.src_tk: Optional[ImageTk.PhotoImage] = None

        self.out_img: Optional[Image.Image] = None
        self.out_tk: Optional[ImageTk.PhotoImage] = None

        self.mesh_img: Optional[Image.Image] = None
        self.mesh_tk: Optional[ImageTk.PhotoImage] = None

        self.result: Optional[HarmonicWarpResult] = None

        self.src_mesh_img: Optional[Image.Image] = None
        self.src_mesh_tk: Optional[ImageTk.PhotoImage] = None

        # 境界条件の「対応点」可視化用
        self.corresp_poly_points: Optional[np.ndarray] = None   # 出力キャンバス座標
        self.corresp_rect_points: Optional[np.ndarray] = None   # 入力画像UV座標
        self.corresp_labels: List[str] = []

        # 直前に挿入した頂点の「削除ガード」
        self._guard_vertex_index: Optional[int] = None
        self._guard_until: float = 0.0

        # 表示用の変換（canvas座標 <-> 画像座標）
        self.in_scale = 1.0
        self.in_offx = 0.0
        self.in_offy = 0.0

        self.out_scale = 1.0
        self.out_offx = 0.0
        self.out_offy = 0.0

        # リサイズ時の再描画を間引く（デバウンス）
        self._redraw_after_id = None

        self._guard_vertex_index: Optional[int] = None
        self._guard_until: float = 0.0

        # ★追加：挿入直後のドラッグ優先ウィンドウ
        self._drag_arm_until: float = 0.0
        self._drag_arm_radius_px: float = 18.0   # 掴みやすさ（表示px）

        # ★追加：一部環境では「ダブルクリック直後のクリック」が <Double-Button-1> として
        # もう一度発火することがある（= クリック列が継続している扱いになる）。
        # そこで、短時間に連続して届いた Double は無視する（デバウンス）。
        # event.time は Tk が付与するミリ秒タイムスタンプ（起点はOS依存）。
        self._last_double_time_ms: int = -10**9
        # Tk が内部で使うダブルクリック判定時間（ms）を拾えればそれを使う。
        # 取得できない環境では無難な値（450ms）にフォールバック。
        try:
            self._double_debounce_ms: int = int(self.tk.call("set", "::tk::Priv(doubleClickTime)"))
        except Exception:
            self._double_debounce_ms = 450

        # 多角形頂点（キャンバス座標）
        self.vertices: List[Tuple[float, float]] = []
        self.closed: bool = False

        # ハンドル（ドラッグ用）
        self.handle_ids: List[int] = []
        self.drag_index: Optional[int] = None

        # ドラッグ操作の種類: None / "vertex" / "segment" / "polygon"
        self.drag_mode: Optional[str] = None
        self.drag_seg_i: Optional[int] = None
        self.drag_seg_j: Optional[int] = None
        self.drag_start_imgxy: Optional[Tuple[float, float]] = None
        # ドラッグ開始時の頂点座標（画像座標）スナップショット
        self.drag_orig_vertices: Optional[np.ndarray] = None

        # 自動クローズ候補（先頭頂点クリック → そのまま離したらクローズ、動かしたら頂点ドラッグ）
        self._close_candidate: bool = False
        self._close_candidate_press_viewxy: Optional[Tuple[float, float]] = None

        # シングルクリック遅延実行ID（ダブルクリックでキャンセル）
        self.single_click_after_id: Optional[str] = None
        self.pending_click_imgxy: Optional[Tuple[float, float]] = None
        self.pending_click_viewxy: Optional[Tuple[float, float]] = None

        # UI構築
        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        # 上部ボタン群
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top, text="画像を開く", command=self.load_image).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(top, text="頂点をクリア", command=self.clear_polygon).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(top, text="1つ戻す", command=self.undo_vertex).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(top, text="ポリゴン保存", command=self.save_polygon).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(top, text="ポリゴン読込", command=self.load_polygon).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(top, text="ワープ実行", command=self.run_warp).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Button(top, text="結果を保存", command=self.save_warped).pack(side=tk.LEFT, padx=10, pady=4)
        tk.Button(top, text="メッシュ画像を保存", command=self.save_mesh).pack(side=tk.LEFT, padx=4, pady=4)

        # パラメータ
        params = tk.Frame(self)
        params.pack(side=tk.TOP, fill=tk.X)

        tk.Label(params, text="max_area（小さいほど高密度）:").pack(side=tk.LEFT, padx=4)
        self.max_area_var = tk.StringVar(value="")
        tk.Entry(params, textvariable=self.max_area_var, width=10).pack(side=tk.LEFT)

        # 放射（一定角度）モード用：角度ステップ（度）
        tk.Label(params, text="放射ステップ(deg):").pack(side=tk.LEFT, padx=10)
        self.ray_step_var = tk.StringVar(value="30.0")
        tk.Entry(params, textvariable=self.ray_step_var, width=6).pack(side=tk.LEFT)

        tk.Label(params, text="min_angle:").pack(side=tk.LEFT, padx=10)
        self.min_angle_var = tk.StringVar(value="28.0")
        tk.Entry(params, textvariable=self.min_angle_var, width=6).pack(side=tk.LEFT)

        # 境界条件モード（既存方式との切替）
        tk.Label(params, text="境界条件:").pack(side=tk.LEFT, padx=10)
        self.boundary_mode_label_var = tk.StringVar(value="4点（四隅）最近点")
        self._boundary_mode_map = {
            "原点(0,0)最近点": "origin",
            "4点（四隅）最近点": "anchors4",
            "8点（四隅+辺中点）最近点": "anchors8",
            "放射（一定角度）交点対応": "rays",
        }
        tk.OptionMenu(
            params,
            self.boundary_mode_label_var,
            *list(self._boundary_mode_map.keys()),
        ).pack(side=tk.LEFT)

        self.overlay_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params, text="メッシュオーバーレイ表示", variable=self.overlay_var, command=self.refresh_output_view)\
            .pack(side=tk.LEFT, padx=12)

        self.show_corresp_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params, text="対応点表示", variable=self.show_corresp_var, command=self._redraw_all_views)\
            .pack(side=tk.LEFT, padx=12)

        self.show_src_mesh_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            params,
            text="元画像メッシュ表示",
            variable=self.show_src_mesh_var,
            command=self.redraw_input
        ).pack(side=tk.LEFT, padx=12)

        # メイン領域：左（入力＆頂点指定）右（出力）
        main = tk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = tk.LabelFrame(main, text="入力（クリックで頂点追加 / ドラッグで移動）")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        right = tk.LabelFrame(main, text="出力")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # キャンバス（サイズは画像読込後に合わせて作る）
        self.in_canvas = tk.Canvas(left, bg="#ddd", highlightthickness=0)
        self.in_canvas.pack(fill=tk.BOTH, expand=True)

        self.out_canvas = tk.Canvas(right, bg="#ddd", highlightthickness=0)
        self.out_canvas.pack(fill=tk.BOTH, expand=True)

        # 下部ヘルプ
        help_ = tk.Label(
            self,
            text="操作: 左クリック=頂点追加 / 頂点=ドラッグ移動 / 線分=ドラッグ移動 / 多角形内部=ドラッグ移動 / 先頭頂点クリックで自動クローズ / ワープ実行",
            anchor="w"
        )
        help_.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)

    def _bind_events(self):
        self.in_canvas.bind("<ButtonPress-1>", self.on_press)
        self.in_canvas.bind("<B1-Motion>", self.on_drag)
        self.in_canvas.bind("<ButtonRelease-1>", self.on_release)
        self.in_canvas.bind("<Double-Button-1>", self.on_double_click)
        self.in_canvas.bind("<Configure>", self.on_canvas_configure)
        self.out_canvas.bind("<Configure>", self.on_canvas_configure)

    # ----------------------------
    #  画像読み込み / 描画
    # ----------------------------

    def load_image(self):
        path = filedialog.askopenfilename(
            title="画像を選択",
            filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGBA")
            self.src_img = img
            self.vertices = []
            self.closed = False
            self.result = None
            self.out_img = None
            self.mesh_img = None
            self.src_mesh_img = None
            self.corresp_poly_points = None
            self.corresp_rect_points = None
            self.corresp_labels = []
            self.redraw_input()
            self.redraw_output()
        except Exception as e:
            messagebox.showerror("エラー", f"画像を読み込めませんでした。\n{e}")

    def _resize_canvases_to_image(self, size: Tuple[int, int]):
        W, H = size
        self.in_canvas.config(width=W, height=H)
        self.out_canvas.config(width=W, height=H)

    def redraw_input(self):
        self.in_canvas.delete("all")
        if self.src_img is None:
            return

        # 表示画像の選択（元画像 or 元画像メッシュ）
        if self.show_src_mesh_var.get() and self.src_mesh_img is not None:
            base = self.src_mesh_img
        else:
            base = self.src_img

        W, H = base.size
        self.in_scale, self.in_offx, self.in_offy = self._compute_view_transform(self.in_canvas, (W, H))

        disp_w = max(1, int(round(W * self.in_scale)))
        disp_h = max(1, int(round(H * self.in_scale)))
        disp = base if self.in_scale == 1.0 else base.resize((disp_w, disp_h), Image.BILINEAR)

        self.src_tk = ImageTk.PhotoImage(disp)
        self.in_canvas.create_image(self.in_offx, self.in_offy, anchor="nw", image=self.src_tk)

        # 元画像メッシュ表示中は編集オーバーレイを出さない（混乱防止）
        if not (self.show_src_mesh_var.get() and self.src_mesh_img is not None):
            self._draw_polygon_on_in_canvas()

        self._draw_correspondences_input()


    def redraw_output(self):
        self.out_canvas.delete("all")
        if self.out_img is None:
            return

        base = self.mesh_img if (self.overlay_var.get() and self.mesh_img is not None) else self.out_img
        W, H = base.size
        self.out_scale, self.out_offx, self.out_offy = self._compute_view_transform(self.out_canvas, (W, H))

        disp_w = max(1, int(round(W * self.out_scale)))
        disp_h = max(1, int(round(H * self.out_scale)))
        disp = base if self.out_scale == 1.0 else base.resize((disp_w, disp_h), Image.BILINEAR)

        self.out_tk = ImageTk.PhotoImage(disp)
        self.out_canvas.create_image(self.out_offx, self.out_offy, anchor="nw", image=self.out_tk)

        self._draw_correspondences_output()


    def refresh_output_view(self):
        self.redraw_output()

    def on_canvas_configure(self, event):
        # 連続リサイズで重くならないようにデバウンス
        if self._redraw_after_id is not None:
            try:
                self.after_cancel(self._redraw_after_id)
            except Exception:
                pass
        self._redraw_after_id = self.after(50, self._redraw_all_views)

    def _redraw_all_views(self):
        self._redraw_after_id = None
        self.redraw_input()
        self.redraw_output()


    # ----------------------------
    #  多角形編集
    # ----------------------------

    def clear_polygon(self):
        self.vertices = []
        self.closed = False
        self.result = None
        self.out_img = None
        self.mesh_img = None
        self.src_mesh_img = None
        self.corresp_poly_points = None
        self.corresp_rect_points = None
        self.corresp_labels = []
        self.redraw_input()
        self.redraw_output()

    def undo_vertex(self):
        if self.closed:
            self.closed = False
        if self.vertices:
            self.vertices.pop()
        self.redraw_input()

    def on_left_click(self, event):
        # ハンドル掴みと衝突するので、closedでないときのみ「追加」とする
        if self.src_img is None:
            return
        if self.closed:
            return
        self.vertices.append((float(event.x), float(event.y)))
        self.redraw_input()

    def _draw_polygon_on_in_canvas(self):
        self.handle_ids = []
        if not self.vertices:
            return

        # 画像座標 -> 表示座標へ変換して描画
        pts_view = []
        for (x, y) in self.vertices:
            vx, vy = self._img_to_view(x, y, self.in_scale, self.in_offx, self.in_offy)
            pts_view.extend([vx, vy])

        if len(self.vertices) >= 2:
            self.in_canvas.create_line(*pts_view, fill="red", width=2)

        if self.closed and len(self.vertices) >= 3:
            x0, y0 = self.vertices[0]
            x1, y1 = self.vertices[-1]
            vx0, vy0 = self._img_to_view(x0, y0, self.in_scale, self.in_offx, self.in_offy)
            vx1, vy1 = self._img_to_view(x1, y1, self.in_scale, self.in_offx, self.in_offy)
            self.in_canvas.create_line(vx1, vy1, vx0, vy0, fill="red", width=2)

        # ハンドル
        for i, (x, y) in enumerate(self.vertices):
            vx, vy = self._img_to_view(x, y, self.in_scale, self.in_offx, self.in_offy)
            r = 5
            hid = self.in_canvas.create_oval(vx - r, vy - r, vx + r, vy + r,
                                             fill="yellow", outline="black", width=1)
            self.in_canvas.itemconfig(hid, tags=(f"handle_{i}", "handle"))
            self.handle_ids.append(hid)

    def _draw_correspondences_input(self):
        """入力キャンバス上に、矩形側（UV）の対応点を描画する。"""
        if not hasattr(self, "show_corresp_var") or not self.show_corresp_var.get():
            return
        if self.src_img is None:
            return
        if self.corresp_rect_points is None or len(self.corresp_rect_points) == 0:
            return

        pts = np.asarray(self.corresp_rect_points, dtype=np.float64)
        labels = self.corresp_labels or [str(i) for i in range(len(pts))]

        for i, (x, y) in enumerate(pts):
            vx, vy = self._img_to_view(float(x), float(y), self.in_scale, self.in_offx, self.in_offy)
            r = 5
            self.in_canvas.create_oval(vx - r, vy - r, vx + r, vy + r, outline="magenta", width=2)
            txt = labels[i] if i < len(labels) else str(i)
            self.in_canvas.create_text(vx + r + 2, vy, text=txt, fill="magenta", anchor="w")

    def _draw_correspondences_output(self):
        """出力キャンバス上に、多角形側（XY）の対応点を描画する。"""
        if not hasattr(self, "show_corresp_var") or not self.show_corresp_var.get():
            return
        if self.out_img is None:
            return
        if self.corresp_poly_points is None or len(self.corresp_poly_points) == 0:
            return

        pts = np.asarray(self.corresp_poly_points, dtype=np.float64)
        labels = self.corresp_labels or [str(i) for i in range(len(pts))]

        for i, (x, y) in enumerate(pts):
            vx, vy = self._img_to_view(float(x), float(y), self.out_scale, self.out_offx, self.out_offy)
            r = 5
            self.out_canvas.create_oval(vx - r, vy - r, vx + r, vy + r, outline="magenta", width=2)
            txt = labels[i] if i < len(labels) else str(i)
            self.out_canvas.create_text(vx + r + 2, vy, text=txt, fill="magenta", anchor="w")

    def _hit_test_handle(self, x_view: float, y_view: float) -> Optional[int]:
        # 表示座標で距離判定（しきい値はpxなので表示座標系が自然）
        for i, (x, y) in enumerate(self.vertices):
            vx, vy = self._img_to_view(x, y, self.in_scale, self.in_offx, self.in_offy)
            if (vx - x_view) ** 2 + (vy - y_view) ** 2 <= 8 ** 2:
                return i
        return None

    def _hit_test_vertex_with_radius(self, idx: int, x_view: float, y_view: float, radius_px: float) -> bool:
        if idx < 0 or idx >= len(self.vertices):
            return False
        x, y = self.vertices[idx]  # 画像座標
        vx, vy = self._img_to_view(x, y, self.in_scale, self.in_offx, self.in_offy)
        r2 = radius_px * radius_px
        return (vx - x_view) ** 2 + (vy - y_view) ** 2 <= r2


    @staticmethod
    def _dist2_point_to_segment(px, py, ax, ay, bx, by) -> Tuple[float, float]:
        """
        点Pと線分ABの距離^2 と、射影パラメータ t（0..1）を返す。
        """
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        vv = vx * vx + vy * vy
        if vv < 1e-12:
            # A==B
            dx = px - ax
            dy = py - ay
            return dx*dx + dy*dy, 0.0
        t = (wx * vx + wy * vy) / vv
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        cx = ax + t * vx
        cy = ay + t * vy
        dx = px - cx
        dy = py - cy
        return dx*dx + dy*dy, t

    def _hit_test_segment(self, x_view: float, y_view: float, thresh_px: float = 6.0) -> Optional[Tuple[int, float, float]]:
        """
        クリック位置に最も近い線分を探す。
        戻り値: (seg_start_index, insert_x, insert_y)
          - seg_start_index: 頂点 i と i+1（または閉路なら最後と最初）を結ぶ線分の開始頂点 index i
          - insert_x/y: 線分への射影点（挿入頂点の初期位置）
        """
        # view -> img
        x, y = self._view_to_img(x_view, y_view, self.in_scale, self.in_offx, self.in_offy)

        n = len(self.vertices)
        if n < 2:
            return None

        best = None
        thresh_img = float(thresh_px) / max(self.in_scale, 1e-9)

        best_d2 = (thresh_img * thresh_img)

        # 対象線分列
        last = n if self.closed else (n - 1)
        for i in range(last):
            j = (i + 1) % n
            ax, ay = self.vertices[i]
            bx, by = self.vertices[j]
            d2, t = self._dist2_point_to_segment(x, y, ax, ay, bx, by)
            if d2 <= best_d2:
                ix = ax + t * (bx - ax)
                iy = ay + t * (by - ay)
                best_d2 = d2
                best = (i, ix, iy)

        return best

    @staticmethod
    def _point_in_polygon(px: float, py: float, poly: np.ndarray) -> bool:
        # 点 (px,py) が多角形内部にあるか（偶奇規則）
        x = poly[:, 0]
        y = poly[:, 1]
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = x[i], y[i]
            xj, yj = x[j], y[j]
            intersect = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _compute_delta_bounds(points: np.ndarray, W: int, H: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # points: (m,2) 画像座標
        dx_min = float(np.max(-points[:, 0]))
        dx_max = float(np.min((W - 1) - points[:, 0]))
        dy_min = float(np.max(-points[:, 1]))
        dy_max = float(np.min((H - 1) - points[:, 1]))
        return (dx_min, dx_max), (dy_min, dy_max)

    @staticmethod
    def _clamp_delta(dx: float, dy: float, dx_bounds: Tuple[float, float], dy_bounds: Tuple[float, float]) -> Tuple[float, float]:
        dx = float(np.clip(dx, dx_bounds[0], dx_bounds[1]))
        dy = float(np.clip(dy, dy_bounds[0], dy_bounds[1]))
        return dx, dy

    def _start_vertex_drag(self, idx: int, x_view: float, y_view: float):
        ix, iy = self._view_to_img(x_view, y_view, self.in_scale, self.in_offx, self.in_offy)
        W, H = self.src_img.size
        ix, iy = self._clamp_xy(ix, iy, W, H)

        self.drag_mode = "vertex"
        self.drag_index = int(idx)
        self.drag_seg_i = None
        self.drag_seg_j = None
        self.drag_start_imgxy = (ix, iy)
        self.drag_orig_vertices = np.asarray(self.vertices, dtype=np.float64).copy()

    def _start_segment_drag(self, i: int, j: int, x_view: float, y_view: float):
        ix, iy = self._view_to_img(x_view, y_view, self.in_scale, self.in_offx, self.in_offy)
        W, H = self.src_img.size
        ix, iy = self._clamp_xy(ix, iy, W, H)

        self.drag_mode = "segment"
        self.drag_index = None
        self.drag_seg_i = int(i)
        self.drag_seg_j = int(j)
        self.drag_start_imgxy = (ix, iy)
        self.drag_orig_vertices = np.asarray(self.vertices, dtype=np.float64).copy()

    def _start_polygon_drag(self, x_view: float, y_view: float):
        ix, iy = self._view_to_img(x_view, y_view, self.in_scale, self.in_offx, self.in_offy)
        W, H = self.src_img.size
        ix, iy = self._clamp_xy(ix, iy, W, H)

        self.drag_mode = "polygon"
        self.drag_index = None
        self.drag_seg_i = None
        self.drag_seg_j = None
        self.drag_start_imgxy = (ix, iy)
        self.drag_orig_vertices = np.asarray(self.vertices, dtype=np.float64).copy()



    def on_press(self, event):
        if self.src_img is None:
            return

        # シングルクリック遅延追加は、ドラッグ開始時に必ずキャンセルする
        self._cancel_pending_single_click()

        # ドラッグ状態を初期化
        self.drag_index = None
        self.drag_mode = None
        self.drag_seg_i = None
        self.drag_seg_j = None
        self.drag_start_imgxy = None
        self.drag_orig_vertices = None

        # 自動クローズ候補を初期化
        self._close_candidate = False
        self._close_candidate_press_viewxy = None

        now = time.monotonic()

        # 挿入直後：クリックが挿入頂点付近ならドラッグ開始を最優先
        if (self._guard_vertex_index is not None) and (now < self._drag_arm_until):
            if self._hit_test_vertex_with_radius(self._guard_vertex_index, event.x, event.y, self._drag_arm_radius_px):
                self._start_vertex_drag(self._guard_vertex_index, event.x, event.y)
                return

        # 頂点ヒット判定
        idx = self._hit_test_handle(event.x, event.y)
        if idx is not None:
            # ポリゴン構築中（closed=False）の「先頭頂点クリック」は自動クローズ候補にする
            if (not self.closed) and (len(self.vertices) >= 3) and (idx == 0):
                self._close_candidate = True
                self._close_candidate_press_viewxy = (float(event.x), float(event.y))

                # ドラッグに転じた場合のために、先頭頂点ドラッグの準備だけ行う
                self._start_vertex_drag(0, event.x, event.y)
                self.drag_mode = None
                self.drag_index = None
                return

            # 通常の頂点ドラッグ
            self._start_vertex_drag(idx, event.x, event.y)
            return

        # 線分ヒット判定：線分をドラッグで並進（両端点を同じΔで移動）
        if len(self.vertices) >= 2:
            hit = self._hit_test_segment(event.x, event.y, thresh_px=6.0)
            if hit is not None:
                seg_i, _, _ = hit
                n = len(self.vertices)
                seg_j = (seg_i + 1) % n if self.closed else (seg_i + 1)
                if 0 <= seg_i < n and 0 <= seg_j < n:
                    self._start_segment_drag(seg_i, seg_j, event.x, event.y)
                    return

        # 多角形全体ドラッグ：閉じた多角形内部を掴んだら全頂点を並進
        if self.closed and len(self.vertices) >= 3:
            ix, iy = self._view_to_img(event.x, event.y, self.in_scale, self.in_offx, self.in_offy)
            W, H = self.src_img.size
            ix, iy = self._clamp_xy(ix, iy, W, H)
            poly = np.asarray(self.vertices, dtype=np.float64)
            if self._point_in_polygon(ix, iy, poly):
                self._start_polygon_drag(event.x, event.y)
                return

        # ここまで来たら、ドラッグ対象ではない → 頂点追加（遅延）
        if self.closed:
            return

        ix, iy = self._view_to_img(event.x, event.y, self.in_scale, self.in_offx, self.in_offy)
        W, H = self.src_img.size
        ix, iy = self._clamp_xy(ix, iy, W, H)

        self.pending_click_imgxy = (ix, iy)
        self.pending_click_viewxy = (float(event.x), float(event.y))
        self.single_click_after_id = self.after(250, self._commit_single_click)

    def on_drag(self, event):
        if self.src_img is None:
            return

        # 自動クローズ候補：動いたら「先頭頂点ドラッグ」に切り替える
        if self._close_candidate and (self._close_candidate_press_viewxy is not None):
            sx, sy = self._close_candidate_press_viewxy
            if (event.x - sx) ** 2 + (event.y - sy) ** 2 >= 4 ** 2:
                self._close_candidate = False
                self._close_candidate_press_viewxy = None
                self._start_vertex_drag(0, sx, sy)

        if self.drag_mode is None:
            return

        # ドラッグ中はシングルクリック追加を確実にキャンセル
        self._cancel_pending_single_click()

        ix, iy = self._view_to_img(event.x, event.y, self.in_scale, self.in_offx, self.in_offy)
        W, H = self.src_img.size
        ix, iy = self._clamp_xy(ix, iy, W, H)

        if self.drag_start_imgxy is None or self.drag_orig_vertices is None:
            return

        x0, y0 = self.drag_start_imgxy
        dx = ix - x0
        dy = iy - y0

        V0 = self.drag_orig_vertices
        Vnew = V0.copy()

        if self.drag_mode == "vertex" and self.drag_index is not None:
            idx = int(self.drag_index)
            pts = V0[[idx], :]
            dx_b, dy_b = self._compute_delta_bounds(pts, W, H)
            dx2, dy2 = self._clamp_delta(dx, dy, dx_b, dy_b)
            Vnew[idx, 0] = V0[idx, 0] + dx2
            Vnew[idx, 1] = V0[idx, 1] + dy2

        elif self.drag_mode == "segment" and (self.drag_seg_i is not None) and (self.drag_seg_j is not None):
            i = int(self.drag_seg_i)
            j = int(self.drag_seg_j)
            pts = V0[[i, j], :]
            dx_b, dy_b = self._compute_delta_bounds(pts, W, H)
            dx2, dy2 = self._clamp_delta(dx, dy, dx_b, dy_b)
            Vnew[i, :] = V0[i, :] + np.array([dx2, dy2], dtype=np.float64)
            Vnew[j, :] = V0[j, :] + np.array([dx2, dy2], dtype=np.float64)

        elif self.drag_mode == "polygon":
            pts = V0
            dx_b, dy_b = self._compute_delta_bounds(pts, W, H)
            dx2, dy2 = self._clamp_delta(dx, dy, dx_b, dy_b)
            Vnew[:, 0] = V0[:, 0] + dx2
            Vnew[:, 1] = V0[:, 1] + dy2

        else:
            return

        self.vertices = [(float(x), float(y)) for x, y in Vnew]

        # ポリゴンが変わったら結果は破棄
        self.result = None
        self.out_img = None
        self.mesh_img = None
        self.src_mesh_img = None
        self.corresp_poly_points = None
        self.corresp_rect_points = None
        self.corresp_labels = []
        self.redraw_input()
        self.redraw_output()

    def on_release(self, event):
        # 自動クローズ候補：押したまま動かず離したらクローズ
        if self._close_candidate:
            self._close_candidate = False
            self._close_candidate_press_viewxy = None
            if (not self.closed) and (len(self.vertices) >= 3):
                self.closed = True
                self.result = None
                self.out_img = None
                self.mesh_img = None
                self.src_mesh_img = None
                self.corresp_poly_points = None
                self.corresp_rect_points = None
                self.corresp_labels = []
                self.redraw_input()
                self.redraw_output()
            return

        self.drag_index = None
        self.drag_mode = None
        self.drag_seg_i = None
        self.drag_seg_j = None
        self.drag_start_imgxy = None
        self.drag_orig_vertices = None

    def on_double_click(self, event):
        #print("on_double_click", event.x, event.y)

        # ★重要：一部環境では「ダブルクリック直後の1クリック」が再び <Double-Button-1> として
        # 発火することがある（例：トリプルクリックの 2-3 回目が Double と判定される）。
        # 連続する Double はここで無視して誤削除/誤挿入を防ぐ。
        # Tk の event.time はミリ秒。
        try:
            t_ms = int(getattr(event, "time", 0))
        except Exception:
            t_ms = 0

        if t_ms != 0:
            if (t_ms - self._last_double_time_ms) < self._double_debounce_ms:
                self.on_press(event)  # ダブルクリック直後のクリックとして扱う
                return "break"
            self._last_double_time_ms = t_ms

        if self.drag_index is not None:
            # 掴めた時点でガード解除してよい（インデックスずれ回避にもなる）
            self._guard_vertex_index = None
            self._guard_until = 0.0
            self._drag_arm_until = 0.0

        if self.src_img is None:
            return

        # ダブルクリックが来たら、シングルクリックによる頂点追加は必ずキャンセル
        self._cancel_pending_single_click()

        now = time.monotonic()

        x = float(event.x)
        y = float(event.y)

        # 1) 頂点上ダブルクリック：頂点削除（ただし挿入直後はガード）
        idx = self._hit_test_handle(x, y)
        if idx is not None:
            if (self._guard_vertex_index == idx) and (now < self._guard_until):
                # 挿入直後の誤削除を防ぐ
                return

            self.vertices.pop(idx)
            if len(self.vertices) < 3:
                self.closed = False

            # 頂点インデックスがずれるのでガードも無効化
            self._guard_vertex_index = None
            self._guard_until = 0.0

            self.result = None
            self.out_img = None
            self.mesh_img = None
            self.src_mesh_img = None
            self.corresp_poly_points = None
            self.corresp_rect_points = None
            self.corresp_labels = []
            self.redraw_input()
            self.redraw_output()
            return


        # 2) 線分上ダブルクリック：その線分に頂点を挿入
        hit = self._hit_test_segment(x, y, thresh_px=6.0)
        if hit is not None:
            seg_i, ix, iy = hit
            n = len(self.vertices)

            # seg_i と seg_i+1 の間に挿入（閉路で最後→最初なら末尾に追加）
            insert_pos = seg_i + 1
            if self.closed and seg_i == n - 1:
                insert_pos = n

            self.vertices.insert(insert_pos, (float(ix), float(iy)))

            now = time.monotonic()

            # 誤削除ガード（既存）
            self._guard_vertex_index = insert_pos
            self._guard_until = now + 0.5

            # ★追加：挿入直後は掴みやすくして、次のクリックでドラッグ開始しやすくする
            self._drag_arm_until = now + 1.0  # 1秒くらいで十分

            self.result = None
            self.out_img = None
            self.mesh_img = None
            self.src_mesh_img = None
            self.corresp_poly_points = None
            self.corresp_rect_points = None
            self.corresp_labels = []
            self.redraw_input()
            self.redraw_output()
            return


        # ヒットしない場合は何もしない（必要ならここで通知）


    def _cancel_pending_single_click(self):
        if self.single_click_after_id is not None:
            try:
                self.after_cancel(self.single_click_after_id)
            except Exception:
                pass
        self.single_click_after_id = None
        self.pending_click_imgxy = None
        self.pending_click_viewxy = None

    def _commit_single_click(self):
        """
        遅延実行される「頂点追加」。
        ダブルクリックが来たらキャンセルされる前提。
        """
        if self.pending_click_imgxy is None or self.pending_click_viewxy is None:
            return
        if self.src_img is None:
            return
        if self.closed:
            return

        x, y = self.pending_click_imgxy
        vx, vy = self.pending_click_viewxy
        # 念のためハンドル近傍なら追加しない（誤追加防止）
        if self._hit_test_handle(vx, vy) is not None:
            return

        # 自動クローズ: 先頭頂点付近をクリックしたら、多角形を閉じる（点は追加しない）
        if (not self.closed) and (len(self.vertices) >= 3):
            x0, y0 = self.vertices[0]
            v0x, v0y = self._img_to_view(x0, y0, self.in_scale, self.in_offx, self.in_offy)
            if (v0x - vx) ** 2 + (v0y - vy) ** 2 <= 12 ** 2:
                self.closed = True
                self._cancel_pending_single_click()
                self.result = None
                self.out_img = None
                self.mesh_img = None
                self.src_mesh_img = None
                self.corresp_poly_points = None
                self.corresp_rect_points = None
                self.corresp_labels = []
                self.redraw_input()
                self.redraw_output()
                return

        self.vertices.append((float(x), float(y)))
        self._cancel_pending_single_click()
        self.result = None
        self.out_img = None
        self.mesh_img = None
        self.src_mesh_img = None
        self.corresp_poly_points = None
        self.corresp_rect_points = None
        self.corresp_labels = []
        self.redraw_input()
        self.redraw_output()

    def _compute_view_transform(self, canvas: tk.Canvas, img_size: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        canvas に img を「収まるように」表示するための (scale, offx, offy) を返す。
        scale は 1.0 を上限（=拡大表示はしない）。
        """
        W, H = img_size
        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())

        scale = min(1.0, cw / W, ch / H)
        disp_w = W * scale
        disp_h = H * scale

        offx = (cw - disp_w) * 0.5
        offy = (ch - disp_h) * 0.5
        return scale, offx, offy

    @staticmethod
    def _img_to_view(x: float, y: float, scale: float, offx: float, offy: float) -> Tuple[float, float]:
        return offx + x * scale, offy + y * scale

    @staticmethod
    def _view_to_img(x: float, y: float, scale: float, offx: float, offy: float) -> Tuple[float, float]:
        return (x - offx) / scale, (y - offy) / scale

    @staticmethod
    def _clamp_xy(x: float, y: float, W: int, H: int) -> Tuple[float, float]:
        x = float(np.clip(x, 0, W - 1))
        y = float(np.clip(y, 0, H - 1))
        return x, y



    # ----------------------------
    #  ワープ実行 / 保存
    # ----------------------------

    def run_warp(self):
        if self.src_img is None:
            messagebox.showwarning("注意", "先に画像を開いてください。")
            return
        if len(self.vertices) < 3:
            messagebox.showwarning("注意", "頂点が3点以上必要です。")
            return
        if not self.closed:
            # 「ワープ実行」時は自動でクローズする。
            # 頂点列は重複点（先頭=末尾）を含めず、閉路であることだけを意味する。
            # build_pslg() は (last -> first) の segment を生成するため、closed フラグだけ立てればよい。
            self.closed = True
            self.redraw_input()

        try:
            # パラメータ
            min_angle = float(self.min_angle_var.get().strip() or "28.0")
            max_area_str = self.max_area_var.get().strip()
            max_area = float(max_area_str) if max_area_str else None

            poly = np.array(self.vertices, dtype=np.float64)
            W, H = self.src_img.size

            # 調和写像ワープ（出力キャンバスは入力画像と同サイズに固定）
            boundary_mode = self._boundary_mode_map.get(self.boundary_mode_label_var.get(), "origin")

            ray_step_deg = float((self.ray_step_var.get() or "30.0").strip() or "30.0")

            res = warp_image_to_polygon_harmonic(
                src_img=self.src_img,
                polygon_xy=poly,
                canvas_size=(W, H),
                max_area=max_area,
                min_angle=min_angle,
                boundary_mode=boundary_mode,
                ray_step_deg=ray_step_deg,
            )
            self.result = res
            self.out_img = res.out_image

            # 対応点（境界条件で用いたアンカー）
            self.corresp_poly_points = getattr(res, 'correspondence_poly_points', None)
            self.corresp_rect_points = getattr(res, 'correspondence_rect_points', None)
            self.corresp_labels = list(getattr(res, 'correspondence_labels', []))

            # メッシュオーバーレイ（ワープ結果を半透明で敷いて、境界太線＋境界点）
            self.mesh_img = render_mesh_overlay_image(
                vertices_xy=res.mesh_vertices_xy,
                triangles=res.triangles,
                segments=res.segments,
                boundary_order=res.boundary_order,
                canvas_size=(W, H),
                base_img=res.out_image,
                base_alpha=0.35,
                edge_width=1,
                boundary_width=3,
                vertex_radius=2,
            )

            # 元画像（入力画像）上にUVメッシュを描いた可視化も作る
            self.src_mesh_img = render_mesh_overlay_image(
                vertices_xy=res.mesh_vertices_uv,   # ★ UVを“座標”として描く
                triangles=res.triangles,
                segments=res.segments,
                boundary_order=res.boundary_order,
                canvas_size=(W, H),                 # 元画像サイズ
                base_img=self.src_img,              # ★ 元画像を背景にする
                base_alpha=1.0,                     # 背景はそのまま表示（半透明にしない）
                edge_width=1,
                boundary_width=3,
                vertex_radius=2,
            )

            self.redraw_input()
            self.redraw_output()

        except Exception as e:
            detail = traceback.format_exc()
            messagebox.showerror("エラー", f"ワープに失敗しました。\n{e}\n\n{detail}")


    def save_polygon(self):
        """現在の多角形（画像座標の頂点列）をファイルに保存する。"""
        if self.src_img is None:
            messagebox.showwarning("注意", "先に画像を開いてください。")
            return
        if len(self.vertices) < 1:
            messagebox.showwarning("注意", "保存する頂点がありません。")
            return

        path = filedialog.asksaveasfilename(
            title="保存（ポリゴン座標）",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return

        W, H = self.src_img.size
        data = {
            "format": "harmonic_warp_polygon",
            "version": 1,
            "image_size": [int(W), int(H)],
            "closed": bool(self.closed),
            "vertices": [[float(x), float(y)] for (x, y) in self.vertices],
            # 参考情報（読込時の動作には必須ではない）
            "boundary_mode_label": self.boundary_mode_label_var.get() if hasattr(self, "boundary_mode_label_var") else "",
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました。\n{e}")

    def load_polygon(self):
        """ファイルから多角形（頂点列）を読み込み、現在の画像に適用する。"""
        if self.src_img is None:
            messagebox.showwarning("注意", "先に画像を開いてください。")
            return

        path = filedialog.askopenfilename(
            title="読込（ポリゴン座標）",
            filetypes=[("JSON", "*.json"), ("Text", "*.txt"), ("All", "*.*")],
        )
        if not path:
            return

        Wc, Hc = self.src_img.size

        try:
            raw = open(path, "r", encoding="utf-8").read()
        except Exception as e:
            messagebox.showerror("エラー", f"ファイルを読み込めませんでした。\n{e}")
            return

        vertices = None
        closed = False
        saved_size = None

        # まず JSON を試す
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "vertices" in obj:
                vertices = obj.get("vertices")
                closed = bool(obj.get("closed", False))
                saved_size = obj.get("image_size", None)
        except Exception:
            obj = None

        # JSONで読めない場合は、簡易テキスト形式（1行に "x y" または "x,y"）を試す
        if vertices is None:
            pts = []
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.replace(",", " ")
                parts = [p for p in line.split() if p]
                if len(parts) >= 2:
                    try:
                        x = float(parts[0]); y = float(parts[1])
                        pts.append([x, y])
                    except Exception:
                        pass
            if pts:
                vertices = pts
                closed = False

        if vertices is None or len(vertices) == 0:
            messagebox.showerror("エラー", "ポリゴン頂点を読み取れませんでした。")
            return

        # サイズが記録されていれば、現在の画像サイズへスケール調整
        sx = sy = 1.0
        if saved_size and isinstance(saved_size, (list, tuple)) and len(saved_size) >= 2:
            try:
                Ws, Hs = float(saved_size[0]), float(saved_size[1])
                if Ws > 1e-9 and Hs > 1e-9 and (int(Ws) != int(Wc) or int(Hs) != int(Hc)):
                    sx = Wc / Ws
                    sy = Hc / Hs
            except Exception:
                sx = sy = 1.0

        # 頂点を現在画像座標に変換＆クランプ
        out = []
        for p in vertices:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            x = float(p[0]) * sx
            y = float(p[1]) * sy
            x, y = self._clamp_xy(x, y, Wc, Hc)
            out.append((x, y))

        if len(out) == 0:
            messagebox.showerror("エラー", "有効な頂点がありません。")
            return

        self.vertices = out
        self.closed = bool(closed) and (len(out) >= 3)

        # 結果は無効化
        self.result = None
        self.out_img = None
        self.mesh_img = None
        self.src_mesh_img = None
        self.corresp_poly_points = None
        self.corresp_rect_points = None
        self.corresp_labels = []

        # ガード状態も解除
        self._guard_vertex_index = None
        self._guard_until = 0.0
        self._drag_arm_until = 0.0

        self.redraw_input()
        self.redraw_output()
    def save_warped(self):
        if self.out_img is None:
            messagebox.showwarning("注意", "先にワープを実行してください。")
            return
        path = filedialog.asksaveasfilename(
            title="保存（ワープ結果）",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if not path:
            return
        try:
            self.out_img.save(path)
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました。\n{e}")

    def save_mesh(self):
        if self.mesh_img is None:
            messagebox.showwarning("注意", "先にワープを実行してください。")
            return
        path = filedialog.asksaveasfilename(
            title="保存（メッシュ可視化）",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if not path:
            return
        try:
            self.mesh_img.save(path)
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました。\n{e}")


if __name__ == "__main__":
    app = HarmonicWarpGUI()
    app.mainloop()
