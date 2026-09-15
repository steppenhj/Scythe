"""청도 산소를 모방한 지형 — 구글어스 실측 경계 반영.

핵심 아이디어 둘:
  1. 지형은 '높이 함수' z(x,y)다. 기울어진 바탕 + 단차 + 봉분 + 경계 둔덕을
     각각 수식으로 만들어 전부 더한다.
  2. 땅의 실제 모양은 '다각형'이다. 구글어스 캡처에서 노란 꼭짓점의 픽셀
     좌표를 그대로 따고, 실측 면적(전체 1,242 ㎡)으로 축척을 보정한다.
     다각형 밖(이웃 밭)은 둔덕으로 한 단 낮다 → 위에서 보면 실제 모양의
     대지가 도드라진다.

배치 (화면 위=북=-x=높은 쪽 가정. 내리막 방향이 반대로 확인되면 TILT_DEG 부호만 뒤집는다):

   [서쪽: 숲 군락(예초 대상 아님)]  [동쪽: 예초 구역 — 봉분 9기, 단차 2]
   예초 구역 꼭짓점마다 흰 말뚝. 경계 밖은 이웃 밭.

실행:  .venv/bin/python terrain.py
마우스 왼쪽 드래그로 시점 회전, 휠로 확대.
"""
import math
import os
import time

# WSLg + Mesa 21.2 조합에서 하드웨어 GL이 조용히 실패한다 → CPU 렌더링 (slope.py 와 동일)
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import mujoco
import mujoco.viewer

# ────── 구글어스 실측 (2026-09-15 캡처) ──────
# 캡처 화면에서 딴 꼭짓점 픽셀 좌표 (가로px, 세로px).
# 새 캡처로 갱신할 땐 이 두 목록과 면적 두 개만 바꾸면 된다.
LAND_PX = [(228, 388), (162, 452), (310, 545), (355, 600), (480, 535), (522, 632),
           (526, 700), (514, 772), (452, 850), (405, 895), (290, 808), (215, 798),
           (140, 785), (52, 745), (18, 604), (95, 517)]
MOW_PX = [(470, 542), (522, 632), (526, 700), (514, 772), (452, 850), (405, 893),
          (350, 846), (312, 786), (302, 720), (315, 650), (345, 595), (395, 560)]
AREA_LAND = 1242.13   # 전체 땅 (㎡) — 축척 보정 기준
AREA_MOW = 547.36     # 예초 구역 (㎡) — 손 트레이싱 오차 보정 기준

# ────── 실측으로 교체할 높이들 (m, 도) — 아직 추정 ──────
TILT_DEG = 3.0        # 전체 기울기 (구글 고도: 전체 고저차 ~3.7 m 에서 역산)
BANK_H = 1.2          # 경계 밖(이웃 밭)으로 내려가는 둔덕 높이
STEPS = [             # 예초 구역 안 단차: (위치 x, 내려가는 높이, 모서리 폭)
    ( 2.0, 0.5, 0.3),
    (12.0, 0.5, 0.3),
]
MOUNDS = [            # 봉분: (x, y, 높이, 반지름) — 위성 배치대로 두 군집
    ( 5.0,  7.5, 0.9, 1.4), ( 5.5, 11.0, 0.8, 1.3), ( 9.0,  6.5, 0.8, 1.3),
    ( 9.5, 10.0, 0.8, 1.3), ( 9.0, 13.5, 0.7, 1.2),
    (14.0,  7.0, 0.8, 1.3), (14.5, 10.5, 0.7, 1.2), (18.0,  8.0, 0.7, 1.2),
    (18.5, 11.5, 0.7, 1.2),
]
TREES = [             # 나무: (x, y, 줄기 반지름, 수관 반지름)
    # 서쪽 숲 군락 — 위성의 짙은 녹색 덩어리
    (-12.0, -15.0, 0.22, 2.8), ( -8.0, -18.0, 0.20, 2.6), ( -4.0, -12.0, 0.18, 2.4),
    (-10.0,  -8.0, 0.19, 2.5), ( -2.0, -18.0, 0.17, 2.3), (  3.0, -14.0, 0.16, 2.2),
    (-15.0, -11.0, 0.18, 2.4), (  1.0,  -7.0, 0.15, 2.0),
    # 예초 구역 안팎의 흩어진 과수 (사진 1)
    ( -1.0,  10.0, 0.15, 2.0), ( -3.0,  17.0, 0.14, 1.9), ( 13.0,  20.0, 0.13, 1.8),
]
# ──────────────────────────────────────────────


def _shoelace(pts):
    """다각형 면적 (신발끈 공식)."""
    p = np.asarray(pts, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


# 픽셀 → 미터: 화면 아래(남)=+x, 오른쪽(동)=+y. 축척은 전체 땅 면적으로 보정
_scale = math.sqrt(AREA_LAND / _shoelace(LAND_PX))
_center = np.asarray(LAND_PX, float).mean(axis=0)
def _to_m(pts):
    p = np.asarray(pts, float)
    return np.stack([(p[:, 1] - _center[1]) * _scale,
                     (p[:, 0] - _center[0]) * _scale], axis=1)

LAND = _to_m(LAND_PX)
MOW = _to_m(MOW_PX)
_shift = (LAND.min(axis=0) + LAND.max(axis=0)) / 2   # 좌표 원점 = 땅 한가운데
LAND -= _shift
MOW -= _shift
# 손 트레이싱 면적 오차(~7%)를 실측 면적에 맞게 소폭 확대 보정
MOW = MOW.mean(axis=0) + (MOW - MOW.mean(axis=0)) * math.sqrt(AREA_MOW / _shoelace(MOW))


def smooth_step(u, w):
    """0→1 로 부드럽게 올라가는 계단. w 가 클수록 모서리가 둥글다(흙이니까)."""
    t = np.clip(u / w, -30, 30)          # exp 오버플로 방지
    return 1.0 / (1.0 + np.exp(-t))


def inside_dist(x, y, poly):
    """다각형 경계까지의 부호 있는 거리: 안이면 +, 밖이면 -."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    inside = np.zeros(np.broadcast(x, y).shape, bool)
    dist = np.full(np.broadcast(x, y).shape, np.inf)
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        # 안/밖: 반직선 교차 횟수의 홀짝 (ray casting)
        cross = (y1 > y) != (y2 > y)
        x_hit = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cross & (x < x_hit)
        # 경계까지 거리: 각 변(선분)까지 최단거리의 최솟값
        dx, dy = x2 - x1, y2 - y1
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy), 0, 1)
        dist = np.minimum(dist, np.hypot(x - x1 - t * dx, y - y1 - t * dy))
    return np.where(inside, dist, -dist)


def height(x, y):
    """묘역의 높이 = 바탕 + 단차들 + 봉분들 + 경계 둔덕의 합."""
    z = -math.tan(math.radians(TILT_DEG)) * np.asarray(x, float)  # ① 기울어진 바탕
    for sx, sh, sw in STEPS:                                      # ② 단차
        z = z - sh * smooth_step(x - sx, sw)
    for mx, my, mh, mr in MOUNDS:                                 # ③ 봉분: 평지에서 바로
        r = np.hypot(x - mx, y - my)                              #    올라오는 둥근 돔
        z = z + mh * (0.5 + 0.5 * np.cos(np.pi * np.minimum(r / mr, 1.0)))
    d = inside_dist(x, y, LAND)                                   # ④ 경계 밖(이웃 밭)은
    z = z - BANK_H * smooth_step(-d, 0.8)                         #    둔덕 아래로
    return z


# 높이 함수를 20 cm 격자로 샘플링 → MuJoCo heightfield 데이터
RX, RY = 28.0, 28.0
NCOL, NROW = 281, 281
X, Y = np.meshgrid(np.linspace(-RX, RX, NCOL), np.linspace(-RY, RY, NROW))
Z = height(X, Y)
ZMIN, ZMAX = float(Z.min()), float(Z.max() - Z.min())


def surf(x, y):
    """월드 좌표에서 지표면의 z (hfield 데이터는 0부터 시작하므로 ZMIN 을 뺀다)."""
    return float(height(x, y)) - ZMIN


# 봉분마다 상석 하나: 봉분 앞(내리막 쪽) 1 m
SANGSEOK = "\n    ".join(
    f'<geom type="box" size=".45 .3 .25" rgba=".6 .6 .62 1" '
    f'pos="{mx + mr + 1.0:.2f} {my:.2f} {surf(mx + mr + 1.0, my) + 0.25:.3f}"/>'
    for mx, my, mh, mr in MOUNDS)

# 나무: 줄기(원기둥)는 부딪히는 장애물, 수관(공)은 충돌 없는 장식
TREE_GEOMS = "\n    ".join(
    f'<geom type="cylinder" size="{tr:.2f} 1.2" rgba=".45 .33 .22 1" '
    f'pos="{tx:.1f} {ty:.1f} {surf(tx, ty) + 1.2:.3f}"/>\n    '
    f'<geom type="sphere" size="{cr:.1f}" rgba=".25 .42 .20 1" contype="0" conaffinity="0" '
    f'pos="{tx:.1f} {ty:.1f} {surf(tx, ty) + 2.4 + cr * 0.5:.3f}"/>'
    for tx, ty, tr, cr in TREES)

# 예초 구역 꼭짓점마다 흰 말뚝 (측량 말뚝처럼 경계 표시)
STAKES = "\n    ".join(
    f'<geom type="cylinder" size=".04 .25" rgba=".95 .95 .9 1" '
    f'pos="{px:.2f} {py:.2f} {surf(px, py) + 0.25:.3f}"/>'
    for px, py in MOW)

XML = f"""
<mujoco>
  <option gravity="0 0 -9.81"/>

  <asset>
    <!-- size = (x반폭, y반폭, 높이 스케일, 바닥 두께). 데이터는 0~1 로 넣는다 -->
    <hfield name="myoyeok" nrow="{NROW}" ncol="{NCOL}" size="{RX} {RY} {ZMAX:.4f} 0.5"/>
    <texture name="grid" type="2d" builtin="checker"
             rgb1=".28 .36 .22" rgb2=".34 .42 .26" width="300" height="300"/>
    <material name="grass" texture="grid" texuniform="true" texrepeat="2 2"/>
  </asset>

  <worldbody>
    <light directional="true" pos="0 0 20" dir="-.2 .2 -1" diffuse=".9 .9 .9"/>

    <geom type="hfield" hfield="myoyeok" material="grass"
          friction="0.6 .005 .0001"/>

    <!-- 상석들: 나중에 '진입 금지 구역'의 기준점이 된다 -->
    {SANGSEOK}

    <!-- 나무들: 줄기는 로봇이 피해야 할 장애물 -->
    {TREE_GEOMS}

    <!-- 예초 구역 경계 말뚝 -->
    {STAKES}

    <!-- 사진 1의 검은 표석 -->
    <geom type="box" size=".3 .08 .28" rgba=".13 .13 .15 1"
          pos="0.0 14.0 {surf(0.0, 14.0) + 0.28:.3f}"/>

    <!-- 크기 감각용 30cm 상자 (slope.py 의 그 상자) -->
    <body pos="16.0 14.0 {surf(16.0, 14.0) + 0.5:.3f}">
      <freejoint/>
      <geom type="box" size=".15 .15 .15" mass="5"
            friction="0.6 .005 .0001" rgba=".8 .3 .2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# ── 아래는 이 파일을 '직접 실행'했을 때만 돈다.
#    field.py 처럼 다른 실험이 위의 지형 재료(XML 조각, surf, Z)를
#    import 로 가져다 쓸 때는 뷰어가 뜨지 않는다.
if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_string(XML)
    model.hfield_data[:] = ((Z - ZMIN) / ZMAX).ravel()   # 격자 데이터를 0~1 로
    data = mujoco.MjData(model)

    print(f"전체 땅 {_shoelace(LAND):.0f} ㎡ ({_shoelace(LAND)/3.3:.0f}평) / "
          f"예초 구역 {_shoelace(MOW):.0f} ㎡ ({_shoelace(MOW)/3.3:.0f}평)")
    print(f"기울기 {TILT_DEG:.0f}도 / 단차 {len(STEPS)}곳 / 경계 둔덕 {BANK_H} m / "
          f"봉분 {len(MOUNDS)}기 / 나무 {len(TREES)}그루")
    print("창을 닫으면 끝납니다.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 예초 구역이 한눈에 보이는 비스듬한 시점에서 시작
        viewer.cam.lookat[:] = [6.0, 8.0, surf(6.0, 8.0)]
        viewer.cam.distance = 42
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -35

        start = time.time()
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            wait = data.time - (time.time() - start)
            if wait > 0:
                time.sleep(wait)
