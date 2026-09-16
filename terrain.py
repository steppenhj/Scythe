"""청도 산소 지형 — 사진에서 뽑은 숫자로 세운 높이함수.

좌표는 terrain_data.py 에서 온다(trace.py 가 사진에서 추출). 이 파일은
그 점·수치를 '땅'으로 바꾸는 일만 한다. 숫자를 여기서 고치지 않는다.

높이 = 기울어진 바탕 + 봉분들 − 경계 밖 둔덕

바탕 기울기는 이제 가정이 아니다. 전체 땅(중앙 219.26 m)과 예초 구역
(중앙 217.82 m)의 도심이 17.8 m 떨어져 있고 1.44 m 차이가 나므로,
내리막은 방위각 120°(동남동), 기울기 4.6도로 역산된다.

예전에 있던 '단차 2개'는 뺐다. 사진에 근거가 없었고, 예초 구역 고저차가
1.66 m 뿐이라 0.5 m 단차 둘이 들어갈 자리가 아니다. 로봇을 세울 단차
기하학은 봉분(지름 2.2 m)이 만든다 — 그게 이 땅의 진짜 요철이다.

실행:  .venv/bin/python terrain.py
마우스 왼쪽 드래그로 시점 회전, 휠로 확대.
"""
import math
import os

# WSLg + Mesa 21.2: 하드웨어 GL이 조용히 실패한다 → CPU 렌더링
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np

import terrain_data as D

# ────── [추정] 사진으로 못 재는 값. 현장 실측으로 교체할 것 ──────
BANK_H = 1.2          # 경계 밖(이웃 밭)으로 내려가는 둔덕 높이
BANK_W = 0.8          # 둔덕이 내려가는 데 걸리는 거리 — 여기가 로봇이 떨어지는 턱
# 상석: 봉분 앞에 놓인 낮은 널돌. 위성으로는 안 잡혀서 일반 치수를 쓴다.
# (전에 쓰던 0.90×0.60×0.50 m 는 봉분 높이의 62% 짜리 덩어리라 과했다 —
#  실제 상석은 정육면체가 아니라 납작한 판이다.)
SANGSEOK_L, SANGSEOK_W, SANGSEOK_H = 1.20, 0.70, 0.25   # 전체 치수 (m)
SANGSEOK_D = 1.8      # 봉분 중심에서 내리막 쪽으로 이만큼 앞
FRICTION = 0.6        # 마른 흙 + 마른 풀
# ────────────────────────────────────────────────────────────

RES = 0.2             # 높이 격자 간격 (m). 화면이 버거우면 여기를 올린다 —
                      #   0.2 → 삼각형 25만개, 봉분 하나를 11칸으로 그린다
                      #   0.3 → 11만개(44%), 7칸    0.4 → 6만개(25%), 6칸
                      # 물리 접촉도 이 격자를 쓰므로, 올리면 봉분 모양이 뭉개진다.
                      # 비교 실험 중에 바꾸면 대조군·실험군이 달라진다 — 먼저 정할 것
MARGIN = 7.0          # 경계 밖으로 더 그릴 여유


def _poly_dist(x, y, poly):
    """다각형 안이면 +, 밖이면 − 로 경계까지의 거리."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    inside = np.zeros(np.shape(x), bool)
    dist = np.full(np.shape(x), np.inf)
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        cross = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_hit = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cross & (x < x_hit)
        dx, dy = x2 - x1, y2 - y1
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy), 0, 1)
        dist = np.minimum(dist, np.hypot(x - x1 - t * dx, y - y1 - t * dy))
    return np.where(inside, dist, -dist)


def _smooth_step(u, w):
    """0 에서 1 로 부드럽게 넘어가는 계단 (폭 w)."""
    t = np.clip(u / w, 0, 1)
    return t * t * (3 - 2 * t)


def height(x, y):
    """지표면 높이. 사진에서 나온 것 + 추정한 둔덕."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    dn = np.asarray(D.DOWNHILL, float)
    z = -D.GRADE * (x * dn[0] + y * dn[1])              # ① 기울어진 바탕
    for mx, my in D.MOUNDS:                             # ② 봉분: 둥근 돔
        r = np.hypot(x - mx, y - my)
        z = z + D.MOUND_H * (0.5 + 0.5 * np.cos(np.pi * np.minimum(r / D.MOUND_R, 1.0)))
    d = np.maximum(_poly_dist(x, y, D.LAND),            # ③ 두 구역의 합집합 밖은
                   _poly_dist(x, y, D.MOW))             #    이웃 밭으로 한 단 아래
    return z - BANK_H * _smooth_step(-d, BANK_W)


# 합집합을 다 담는 격자
_ax = [p[0] for p in D.LAND + D.MOW]
_ay = [p[1] for p in D.LAND + D.MOW]
RX = max(abs(min(_ax)), abs(max(_ax))) + MARGIN
RY = max(abs(min(_ay)), abs(max(_ay))) + MARGIN
NCOL = int(2 * RX / RES) | 1
NROW = int(2 * RY / RES) | 1
X, Y = np.meshgrid(np.linspace(-RX, RX, NCOL), np.linspace(-RY, RY, NROW))
Z = height(X, Y)
ZMIN, ZMAX = float(Z.min()), float(Z.max() - Z.min())


def surf(x, y):
    """월드 좌표에서 지표면 z. hfield 데이터가 0 부터 시작하므로 ZMIN 을 뺀다."""
    return float(height(x, y)) - ZMIN


def _inside_union(x, y):
    return (_poly_dist(x, y, D.LAND) > 0) or (_poly_dist(x, y, D.MOW) > 0)


# 검산: 모델이 만든 기복이 구글 고도와 같은 자릿수인가
_in = (_poly_dist(X, Y, D.LAND) > 0) | (_poly_dist(X, Y, D.MOW) > 0)
RELIEF = float(Z[_in].max() - Z[_in].min())

# ────── 지형 위에 얹는 것들 ──────
_dn = np.asarray(D.DOWNHILL, float)

# 상석: 봉분 앞(내리막 쪽). 내리막 방향을 알아서 이제 제대로 놓인다.
# size 는 반치수라 절반씩 넣는다. 로봇에겐 넘을 수 없는 장애물이자 진입 금지 표식.
def _sangseok_xml(mx, my):
    x, y = mx + _dn[0] * SANGSEOK_D, my + _dn[1] * SANGSEOK_D
    return (f'<geom type="box" '
            f'size="{SANGSEOK_L/2:.3f} {SANGSEOK_W/2:.3f} {SANGSEOK_H/2:.3f}" '
            f'rgba=".6 .6 .62 1" '
            f'pos="{x:.2f} {y:.2f} {surf(x, y) + SANGSEOK_H/2:.3f}"/>')

SANGSEOK = "\n    ".join(_sangseok_xml(mx, my) for mx, my in D.MOUNDS)

# 나무. 경계 안이면 줄기에 충돌이 있고(로봇이 피해야 한다), 밖이면 배경.
# 숲 덩어리는 trace.py 가 이미 개별 나무로 쪼개서 준다 — 여기서는 한 그루씩만 그린다.
def _tree_xml(tx, ty, rad, obstacle):
    z = surf(tx, ty)
    trunk_r = max(0.10, rad * 0.11)
    col = "" if obstacle else ' contype="0" conaffinity="0"'
    return (f'<geom type="cylinder" size="{trunk_r:.2f} 1.2"{col} rgba=".45 .33 .22 1" '
            f'pos="{tx:.1f} {ty:.1f} {z + 1.2:.2f}"/>\n    '
            f'<geom type="sphere" size="{rad:.2f}" contype="0" conaffinity="0" '
            f'rgba=".25 .42 .20 1" pos="{tx:.1f} {ty:.1f} {z + 2.4 + rad*0.5:.2f}"/>')

TREE_GEOMS = "\n    ".join(_tree_xml(*t) for t in D.TREES)

# 예초 구역 경계 말뚝 — 로봇이 넘으면 안 되는 선
STAKES = "\n    ".join(
    f'<geom type="cylinder" size=".04 .25" rgba=".95 .95 .9 1" '
    f'pos="{px:.2f} {py:.2f} {surf(px, py) + 0.25:.3f}"/>'
    for px, py in D.MOW)

XML = f"""
<mujoco>
  <option gravity="0 0 -9.81"/>

  <!-- CPU 렌더링(WSLg) 예산에 맞춘 화질. 물리에는 아무 영향이 없다.
       offsamples 0: 오프스크린 멀티샘플 끔 (기본 4 = 화소당 4배 일)
       numslices/numstacks: 구·원기둥 세분. 28x16 → 12x8 이면 수관 삼각형이 1/5 -->
  <visual>
    <quality shadowsize="1024" offsamples="0" numslices="12" numstacks="8"/>
  </visual>

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
          friction="{FRICTION} .005 .0001"/>

    {SANGSEOK}
    {TREE_GEOMS}
    {STAKES}
  </worldbody>
</mujoco>
"""

if __name__ == "__main__":
    import time
    import mujoco
    import mujoco.viewer

    print(__doc__)
    print(f"격자   {NCOL}×{NROW} @ {RES} m,  범위 ±{RX:.1f} × ±{RY:.1f} m")
    print(f"봉분   {len(D.MOUNDS)}기, 지름 {2*D.MOUND_R:.1f} m, 높이 {D.MOUND_H} m [추정]")
    print(f"나무   {len(D.TREES)}덩어리 (경계 안 장애물 {sum(1 for t in D.TREES if t[3])}개)")
    print(f"내리막 방위각 {(math.degrees(math.atan2(*D.DOWNHILL))+360)%360:.0f}°, "
          f"{math.degrees(math.atan(D.GRADE)):.1f}도")
    print(f"검산   모델 기복 {RELIEF:.2f} m  vs  구글 고저차 {D.Z_RANGE:.2f} m "
          f"(차이 {abs(RELIEF-D.Z_RANGE):.2f} m)")

    model = mujoco.MjModel.from_xml_string(XML)
    model.hfield_data[:] = ((Z - ZMIN) / ZMAX).ravel()
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 70
        viewer.cam.elevation = -35
        # 움직이는 게 없는 정적인 장면이다. 물리를 돌릴 것도 없고,
        # viewer.sync() 는 렌더 스레드와 같은 락을 잡으므로 성기게 부른다.
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 20)
