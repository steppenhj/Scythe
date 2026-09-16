"""사진 → 숫자. 구글어스 캡처에서 지형 재료를 뽑아 terrain_data.py 를 쓴다.

terrain.py 가 이 결과를 읽는다. 사진을 새로 뜨면 photos/ 를 갈아끼우고
이 파일을 다시 돌리면 된다 — 손으로 좌표를 고쳐 적지 않는다.

  .venv/bin/python trace.py

출처 등급을 셋으로 나눠서 표시한다. 이게 이 파일의 핵심이다:
  [측정] 구글어스가 화면에 찍어준 값 (면적, 둘레, 고도)
  [추출] 사진 픽셀에서 기계적으로 뽑은 값 (다각형 꼭짓점, 정합 이동량)
  [손]   사진을 보고 사람이 찍은 값 (봉분 위치) — 재확인은 되지만 자동은 아니다
  [추정] 사진에 없는 값 (봉분 높이, 마찰) — 현장 실측으로만 맞춘다
"""
import math
from collections import deque

import numpy as np
from PIL import Image

PHOTOS = "photos/"
LAND_IMG = PHOTOS + "전체사진_여길_다_예초하는건아님.png"
MOW_IMG = PHOTOS + "위치위성사진_확실하진 않음.png"

# ────── [측정] 구글어스 측정 패널에서 읽은 값 ──────
AREA_LAND, PERIM_LAND = 1242.13, 159.41      # m², m
AREA_MOW, PERIM_MOW = 547.36, 103.68
ELEV_LAND = (217.48, 219.26, 221.13)          # 최소 / 중앙 / 최대 (m)
ELEV_MOW = (216.94, 217.82, 218.60)

# ────── [손] 봉분 중심, '전체사진' 픽셀 좌표 ──────
# 6배 확대 + 국소대비 영상에서 읽었다. 각 봉분은 북쪽에 그림자 모자,
# 남쪽에 햇빛 받는 밝은 면이 있다 — 그 둘의 경계가 마루다.
# 자동 검출은 숲의 대비에 밀려 실패했다(정합필터·임계값 둘 다). 그래서 손.
MOUND_PX = [(377, 703), (372, 738), (356, 784), (428, 786),
            (350, 811), (405, 822), (392, 854)]
MOUND_R = 1.1        # [추출] 확대 영상에서 잰 봉분 반지름 (m), 지름 ~2.2 m
MOUND_H = 0.8        # [추정] 사진으로는 높이를 못 잰다. 현장에서 잴 것

TREE_NEAR_PX = 70    # 경계에서 이만큼 밖까지의 나무는 배경으로 같이 담는다


# ══════════ 사진에서 뽑기 ══════════
def _load(path):
    return np.asarray(Image.open(path).convert("RGB"), float)


def _flood(mask, seed):
    h, w = mask.shape
    out = np.zeros_like(mask)
    q = deque([seed])
    out[seed] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                q.append((ny, nx))
    return out


def _roof_box(im):
    """두 사진에 공통으로 찍힌 초록 지붕 창고 → 정합 기준점."""
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    m = (b > r + 35) & (g > r + 30) & (r < 80) & (60 < b) & (b < 130)
    seed = next((y, x) for y in range(im.shape[0] // 3)
                for x in range(600, 1000) if m[y, x])
    ys, xs = np.nonzero(_flood(m, seed))
    return xs.min(), xs.max(), ys.min(), ys.max()


def register(a, b):
    """a 좌표 + (dx,dy) = b 좌표. 배율도 같이 검산해서 돌려준다."""
    ax0, ax1, ay0, ay1 = _roof_box(a)
    bx0, bx1, by0, by1 = _roof_box(b)
    sx, sy = (bx1 - bx0) / (ax1 - ax0), (by1 - by0) / (ay1 - ay0)
    return (bx0 - ax0, by0 - ay0), (sx, sy)


def _yellow(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > 170) & (g > 130) & (b < 110) & (r - b > 80) & (g - b > 50)


def _blobs(mask, minpix):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        if len(pts) >= minpix:
            a = np.asarray(pts, float)
            out.append((a[:, 1].mean(), a[:, 0].mean()))
    return out


def polygon(im):
    """노란 측량 다각형의 꼭짓점을, 그려진 순서(사이클)대로 뽑는다.

    꼭짓점 마커는 노란 테두리 안의 흰 원이다. 구글어스는 변의 중점에도
    작은 점을 찍으므로(끌어서 꼭짓점을 늘리는 손잡이) 크기로 걸러낸다.
    """
    ym = _yellow(im)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    near = np.zeros_like(ym)
    ys, xs = np.nonzero(ym)
    for dy in range(-9, 10):
        for dx in range(-9, 10):
            yy, xx = ys + dy, xs + dx
            k = (yy >= 0) & (yy < ym.shape[0]) & (xx >= 0) & (xx < ym.shape[1])
            near[yy[k], xx[k]] = True
    V = _blobs((r > 205) & (g > 205) & (b > 195) & near, 100)   # 큰 원만 = 진짜 꼭짓점

    def on_edge(i, j):
        """i-j 가 다각형의 변인가: 선분이 노란 선을 따라가고, 제3의 꼭짓점을 관통하지 않는다."""
        (x1, y1), (x2, y2) = V[i], V[j]
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        for k in range(len(V)):
            if k in (i, j):
                continue
            xk, yk = V[k]
            t = max(0.0, min(1.0, ((xk - x1) * dx + (yk - y1) * dy) / L))
            if math.hypot(xk - x1 - t * dx, yk - y1 - t * dy) < 10:
                return False
        n = int(max(abs(dx), abs(dy)))
        if n < 8:
            return False
        hit = tot = 0
        for s in range(n + 1):
            t = s / n
            if t < 0.12 or t > 0.88:        # 마커 반지름 회피
                continue
            x, y = x1 + dx * t, y1 + dy * t
            tot += 1
            if ym[max(0, int(y) - 6):int(y) + 7, max(0, int(x) - 6):int(x) + 7].any():
                hit += 1
        return tot > 0 and hit / tot > 0.97

    adj = {i: [] for i in range(len(V))}
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if on_edge(i, j):
                adj[i].append(j)
                adj[j].append(i)
    start = min(range(len(V)), key=lambda i: V[i][1])
    cyc, prev, cur = [start], -1, start
    while True:
        nxt = [k for k in adj[cur] if k != prev]
        if not nxt:
            break
        k = min(nxt, key=lambda k: math.hypot(V[k][0] - V[cur][0], V[k][1] - V[cur][1]))
        if k == start:
            break
        cyc.append(k)
        prev, cur = cur, k
        if len(cyc) > len(V):
            raise RuntimeError("다각형 사이클이 닫히지 않는다")
    if len(cyc) != len(V):
        raise RuntimeError(f"꼭짓점 {len(V)}개 중 {len(cyc)}개만 사이클에 들어갔다")
    return [V[i] for i in cyc]


def canopy(im, land_px, mow_px):
    """수관 = 어둡고 푸른끼 도는 덩어리. 경계 안이면 장애물, 밖이면 배경."""
    v = im.mean(2)
    r, b = im[..., 0], im[..., 2]
    h, w = v.shape
    gy, gx = np.mgrid[0:h, 0:w]
    pt = np.stack([gx, gy], -1).astype(float)
    u = inside(pt, land_px) | inside(pt, mow_px)
    near = u.copy()                      # 경계 밖 TREE_NEAR_PX 까지 넓힌다
    for _ in range(TREE_NEAR_PX):
        near[1:] |= near[:-1]; near[:-1] |= near[1:]
        near[:, 1:] |= near[:, :-1]; near[:, :-1] |= near[:, 1:]
    out = []
    for cx, cy, n in _blobs2(near & (v < 92) & (b >= r - 10), 80):
        rad = math.sqrt(n / math.pi)
        out.append((cx, cy, rad, bool(u[int(cy), int(cx)])))
    return out


def _blobs2(mask, minpix):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    res = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)]); seen[y0, x0] = True; pts = []
        while q:
            y, x = q.popleft(); pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; q.append((ny, nx))
        if len(pts) >= minpix:
            a = np.asarray(pts, float)
            res.append((a[:, 1].mean(), a[:, 0].mean(), len(pts)))
    return res


def shoelace(poly):
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def perimeter(poly):
    p = np.asarray(poly, float)
    return float(np.hypot(*(np.roll(p, -1, 0) - p).T).sum())


def centroid(poly):
    """면적 가중 도심 (꼭짓점 평균이 아니다 — 변 길이가 들쭉날쭉하다)."""
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y2 - x2 * y
    A = cr.sum() / 2
    return np.array([((x + x2) * cr).sum() / (6 * A), ((y + y2) * cr).sum() / (6 * A)])


def inside(pts, poly):
    poly = np.asarray(poly, float)
    x, y = pts[..., 0], pts[..., 1]
    ins = np.zeros(x.shape, bool)
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        cross = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xh = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        ins ^= cross & (x < xh)
    return ins


# ══════════ 실행 ══════════
def main():
    A, B = _load(MOW_IMG), _load(LAND_IMG)
    (dx, dy), (sx, sy) = register(A, B)
    print(f"[추출] 정합: 예초 사진 + ({dx:+.0f},{dy:+.0f}) = 전체 사진,  배율 {sx:.4f}×{sy:.4f}")
    if abs(sx - 1) > 0.01 or abs(sy - 1) > 0.01:
        raise RuntimeError("두 캡처의 줌이 다르다 — 평행이동만으로는 못 맞춘다")

    land_px = polygon(B)
    mow_px = [(x + dx, y + dy) for x, y in polygon(A)]   # 전체 사진 좌표계로
    print(f"[추출] 꼭짓점: 전체 땅 {len(land_px)}개, 예초 구역 {len(mow_px)}개")

    # 축척: 두 다각형이 각자 독립으로 cm/px 를 준다 → 서로 검산이 된다
    s_land = math.sqrt(AREA_LAND / shoelace(land_px))
    s_mow = math.sqrt(AREA_MOW / shoelace(mow_px))
    print(f"[검산] 축척: 전체 {s_land*100:.3f} cm/px, 예초 {s_mow*100:.3f} cm/px "
          f"→ {abs(s_mow/s_land-1)*100:.1f}% 차이")
    if abs(s_mow / s_land - 1) > 0.03:
        raise RuntimeError("두 축척이 3% 넘게 어긋난다 — 트레이싱이나 면적을 의심할 것")
    S = (s_land + s_mow) / 2

    # 둘레로 한 번 더 검산 (면적과 독립인 양이다)
    print(f"[검산] 둘레: 전체 {perimeter(land_px)*S:6.2f} m (구글 {PERIM_LAND}), "
          f"예초 {perimeter(mow_px)*S:6.2f} m (구글 {PERIM_MOW})")

    # 원점 = 두 다각형 합집합의 도심. +x 동쪽, +y 북쪽 (사진 y 는 남쪽이라 뒤집는다)
    org = (centroid(land_px) * AREA_LAND + centroid(mow_px) * AREA_MOW) / (AREA_LAND + AREA_MOW)
    to_m = lambda pts: [((x - org[0]) * S, -(y - org[1]) * S) for x, y in pts]
    LAND, MOW, MOUNDS = to_m(land_px), to_m(mow_px), to_m(MOUND_PX)

    # 겹침: 예초 구역이 전체 땅 밖으로 얼마나 나가나
    g = np.mgrid[0:B.shape[0]:1, 0:B.shape[1]:1]
    PT = np.stack([g[1], g[0]], -1).astype(float)
    im_, il_ = inside(PT, land_px), inside(PT, mow_px)
    out = (il_ & ~im_).sum() / il_.sum()
    print(f"[추출] 예초 구역의 {out*100:.1f}% 가 전체 땅 다각형 밖 "
          f"({(il_&~im_).sum()*S*S:.0f} m²) → 합집합을 작업 범위로 쓴다")

    # 기울기 방향: 두 다각형의 도심 위치와 중앙 고도 차이에서 역산.
    # 지금까지는 '북쪽이 높다'고 가정만 했는데, 측정값이 방향을 준다.
    cl, cm = centroid(land_px), centroid(mow_px)
    v = np.array([(cm[0] - cl[0]) * S, -(cm[1] - cl[1]) * S])   # 전체→예초 (m, 동/북)
    drop = ELEV_MOW[1] - ELEV_LAND[1]                            # 음수 = 예초가 낮다
    dist = float(np.hypot(*v))
    grade = -drop / dist
    down = v / dist
    bearing = (math.degrees(math.atan2(down[0], down[1])) + 360) % 360
    print(f"[측정] 중앙 고도: 전체 {ELEV_LAND[1]} m, 예초 {ELEV_MOW[1]} m → {drop:+.2f} m")
    print(f"[추출] 내리막 방향 방위각 {bearing:.0f}° (0=북,90=동), "
          f"도심 간 {dist:.1f} m 에 {-drop:.2f} m 강하 = {math.degrees(math.atan(grade)):.1f}°")

    zr_land = ELEV_LAND[2] - ELEV_LAND[0]
    zr_mow = ELEV_MOW[2] - ELEV_MOW[0]
    zr_union = max(ELEV_LAND[2], ELEV_MOW[2]) - min(ELEV_LAND[0], ELEV_MOW[0])
    print(f"[측정] 고저차: 전체 {zr_land:.2f} m, 예초 {zr_mow:.2f} m, 합집합 {zr_union:.2f} m")

    trees = canopy(B, land_px, mow_px)
    n_in = sum(1 for *_, ins in trees if ins)
    print(f"[추출] 수관 덩어리 {len(trees)}개 — 경계 안 {n_in}개(장애물), "
          f"밖 {len(trees)-n_in}개(배경). 서쪽 '숲 군락'은 경계 밖이다")
    TREES = [(((cx - org[0]) * S), (-(cy - org[1]) * S), rad * S, ins)
             for cx, cy, rad, ins in trees]

    W = lambda pts: "[\n" + "".join(f"    ({x:8.3f}, {y:8.3f}),\n" for x, y in pts) + "]"
    W3 = ("[\n" + "".join(f"    ({x:8.3f}, {y:8.3f}, {r:5.2f}, {ins}),\n"
                          for x, y, r, ins in TREES) + "]")
    with open("terrain_data.py", "w") as f:
        f.write(f'''"""trace.py 가 사진에서 뽑아 쓴 파일. 손으로 고치지 말 것.

다시 만들려면:  .venv/bin/python trace.py

좌표계: 원점은 전체 땅 ∪ 예초 구역의 면적가중 도심, +x 동쪽, +y 북쪽 (m).
출처 등급은 trace.py 의 docstring 참고.
"""
# [추출] 사진 픽셀 → m
SCALE = {S:.6f}              # m/px
LAND = {W(LAND)}
MOW = {W(MOW)}
MOUNDS = {W(MOUNDS)}          # [손] 중심만. 반지름·높이는 아래 스칼라

# [추출] (x, y, 수관반지름 m, 경계 안인가). 안이면 로봇이 부딪히는 장애물
TREES = {W3}

# [추출] 확대 영상에서 잰 값
MOUND_R = {MOUND_R}
# [추정] 사진으로 못 재는 값 — 현장 실측 전까지 임시
MOUND_H = {MOUND_H}

# [측정] 구글어스 패널
AREA_LAND, AREA_MOW = {AREA_LAND}, {AREA_MOW}
ELEV_LAND = {ELEV_LAND}       # 최소 / 중앙 / 최대 (m)
ELEV_MOW = {ELEV_MOW}
Z_RANGE = {zr_union:.2f}                # 합집합 고저차 (m)

# [추출] 두 다각형의 도심·고도 차이에서 역산한 내리막
DOWNHILL = ({down[0]:.4f}, {down[1]:.4f})   # 단위벡터 (동, 북)
GRADE = {grade:.5f}                # 기울기 (tan). {math.degrees(math.atan(grade)):.2f}도
''')
    print("\n→ terrain_data.py 갱신 완료")


if __name__ == "__main__":
    main()
