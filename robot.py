"""Phase 0: 궤도 차체가 평지에서 조종대로 움직이나.

첫 로봇 = 몸통 상자 하나 + 궤도 판 두 개. 진짜 벨트와 바퀴 수십 개를
만드는 대신, MuJoCo 의 surfacevel 로 궤도 접촉면을 컨베이어처럼 민다.
(궤도가 땅을 미는 원리 자체가 '접촉면의 마찰'이라 흉내로 충분하다.
 함정: XML 에 0 이 아닌 surfacevel 을 선언해 둬야 런타임 쓰기가 먹는다.)

조종 (뷰어 창을 클릭한 뒤):
  ↑/W 속도 한 칸 올림   ↓/S 한 칸 내림   (크루즈처럼 유지된다)
  ←/A →/D 조향 — 놓으면 저절로 직진으로 복귀   스페이스/X 정지

화면이 느리면 (물리는 이미 실시간의 46배로 돌 수 있다 — 병목은 항상 렌더링이다):
  1. 뷰어 왼쪽 Option 패널에서 **Vertical Sync 끄기** — 클릭 한 번, 파일 수정 불필요
  2. SHADOW 를 0 으로 (그림자 끔). 입체감은 줄지만 가장 크게 빨라진다
  3. RENDER_HZ 를 30, 20 으로
  4. terrain.py 의 RES 를 0.3 으로 (삼각형 25만 → 11만).
     단 물리 접촉도 이 격자를 쓴다 — 비교 실험 중에 바꾸면 대조군이 달라진다

실행:  .venv/bin/python robot.py
"""
import math
import os
import time

# WSLg + Mesa 21.2: 하드웨어 GL이 조용히 실패한다 → CPU 렌더링
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import mujoco
import mujoco.viewer

# ────── 차체 가설 (A안 30~40kg 급) — 바꿔가며 실험 ──────
BODY_L = 0.80     # 몸통 길이 (m)
BODY_W = 0.50     # 몸통 폭 — 궤도까지 합친 전폭은 0.80
BODY_H = 0.20     # 몸통 높이
TRACK_W = 0.15    # 궤도 판 폭
TRACK_H = 0.12    # 궤도 판 높이 (지상고는 이 절반쯤)
BODY_M, TRACK_M = 25.0, 5.0   # 질량: 25 + 5×2 = 35 kg
FRICTION = 0.6
MAX_V = 1.0       # 최고 궤도 속도 (m/s) — '이동은 빠르게'의 상한 후보
STEP_V = 0.25     # 키 한 번에 바뀌는 속도
STEER_V = 0.5     # 조향 키 한 번의 세기
STEER_TAU = 0.6   # 조향이 직진으로 복귀하는 시간 상수 (초) — 핸들 놓으면 돌아오듯
RENDER_HZ = 50    # 화면 갱신 (Hz). WSLg CPU 렌더링이 버거우면 30, 20 으로 내린다.
                  # 물리는 이것과 무관하게 2 ms 마다 푼다 — 화면만 성기어진다
SHADOW = 1024     # 그림자맵 한 변 (px). MuJoCo 기본 4096 은 CPU 렌더링에 과하다 —
                  # 조명 하나당 16.8 M 화소를 프레임마다 따로 그린다. 0 이면 그림자 끔
# ──────────────────────────────────────────────────────

XML = f"""
<mujoco>
  <!-- cone="elliptic": 마찰 원뿔을 기본 피라미드 근사 대신 정확한 원뿔로.
       스키드 스티어는 회전할 때 마찰 방향이 비스듬해서 근사 오차가 크게 보인다 -->
  <option gravity="0 0 -9.81" cone="elliptic"/>

  <!-- CPU 렌더링(WSLg) 예산에 맞춘 화질. 물리에는 아무 영향이 없다.
       offsamples 0: 오프스크린 멀티샘플 끔 (기본 4 = 화소당 4배 일)
       numslices/numstacks: 구·원기둥 세분. 28x16 → 12x8 이면 수관 삼각형이 1/5 -->
  <visual>
    <quality shadowsize="{SHADOW}" offsamples="0" numslices="12" numstacks="8"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1=".28 .36 .22" rgb2=".34 .42 .26" width="300" height="300"/>
    <material name="grass" texture="grid" texuniform="true" texrepeat="2 2"/>
  </asset>

  <worldbody>
    <light directional="true" pos="0 0 10" dir="-.2 .2 -1" diffuse=".9 .9 .9"/>
    <geom name="ground" type="plane" size="60 60 .1"
          friction="{FRICTION} .005 .0001" material="grass"/>

    <body name="robot" pos="0 0 {TRACK_H / 2 + 0.05:.3f}">
      <freejoint/>
      <!-- 몸통: 궤도 위에 얹힌 상자 -->
      <geom name="chassis" type="box"
            size="{BODY_L / 2} {BODY_W / 2} {BODY_H / 2}"
            pos="0 0 {TRACK_H / 2 + BODY_H / 2:.3f}"
            mass="{BODY_M}" rgba=".8 .3 .2 1"/>
      <!-- 궤도 판 두 개: 땅에 닿는 건 이 둘뿐.
           surfacevel 을 0 아닌 값으로 선언해야 런타임 쓰기가 먹는다 -->
      <geom name="ltrack" type="box"
            size="{BODY_L / 2} {TRACK_W / 2} {TRACK_H / 2}"
            pos="0 {(BODY_W + TRACK_W) / 2:.3f} 0" mass="{TRACK_M}"
            friction="{FRICTION} .005 .0001" surfacevel="0.0001 0 0"
            rgba=".15 .15 .15 1"/>
      <geom name="rtrack" type="box"
            size="{BODY_L / 2} {TRACK_W / 2} {TRACK_H / 2}"
            pos="0 {-(BODY_W + TRACK_W) / 2:.3f} 0" mass="{TRACK_M}"
            friction="{FRICTION} .005 .0001" surfacevel="0.0001 0 0"
            rgba=".15 .15 .15 1"/>
      <!-- castshadow false: 그림자맵은 태양(위 directional) 것 하나면 된다 -->
      <light mode="trackcom" pos="0 0 4" dir="0 0 -1" diffuse=".8 .8 .8"
             castshadow="false"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

LT = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ltrack")
RT = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rtrack")
assert model.flg_surfacevel, "surfacevel 플래그가 꺼져 있다 (XML 선언 확인)"

# 조종 상태: v = 전진 속도, w = 좌우 차동 (좌회전 +)
cmd = {"v": 0.0, "w": 0.0}

def key_callback(key):
    if key in (265, ord("W")):            # ↑
        cmd["v"] += STEP_V
    elif key in (264, ord("S")):          # ↓
        cmd["v"] -= STEP_V
    elif key in (263, ord("A")):          # ←
        cmd["w"] = min(cmd["w"] + STEER_V, MAX_V)
    elif key in (262, ord("D")):          # →
        cmd["w"] = max(cmd["w"] - STEER_V, -MAX_V)
    elif key in (32, ord("X")):           # 스페이스
        cmd["v"] = cmd["w"] = 0.0
    cmd["v"] = max(-MAX_V, min(MAX_V, cmd["v"]))
    cmd["w"] = max(-MAX_V, min(MAX_V, cmd["w"]))   # 상태는 메인 루프가 찍는다

def apply_tracks():
    """궤도 표면 속도 쓰기. 표면이 뒤(-x)로 돌아야 차체가 앞(+x)으로 간다."""
    vl = cmd["v"] - cmd["w"]              # 좌회전이면 왼쪽이 느리다
    vr = cmd["v"] + cmd["w"]
    model.geom_surfacevel[LT][0] = -vl
    model.geom_surfacevel[RT][0] = -vr

def yaw_of(q):
    """쿼터니언 → 진행 방향(요) 각도, 도 단위."""
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

print(__doc__)
print(f"물리 {model.opt.timestep*1000:.0f} ms/스텝, 화면 {RENDER_HZ} Hz "
      f"→ 한 프레임에 {max(1, round(1/RENDER_HZ/model.opt.timestep))} 스텝")

with mujoco.viewer.launch_passive(model, data,
                                  key_callback=key_callback) as viewer:
    viewer.cam.distance = 4.0
    viewer.cam.elevation = -20

    steer_decay = math.exp(-model.opt.timestep / STEER_TAU)
    cam_decay = math.exp(-model.opt.timestep / 0.7)
    cam_az = yaw_of(data.qpos[3:7])
    steps = max(1, round(1 / RENDER_HZ / model.opt.timestep))

    start = time.time()
    last_print = 0.0
    while viewer.is_running():

        # 물리를 한 프레임치 몰아서 푼 뒤에 화면을 한 번 갱신한다
        for _ in range(steps):
            cmd["w"] *= steer_decay       # 핸들에서 손 떼면 직진 복귀
            apply_tracks()
            mujoco.mj_step(model, data)

        # 카메라: 위치는 로봇을 따라가고, 방향은 천천히 따라온다
        diff = (yaw_of(data.qpos[3:7]) - cam_az + 180) % 360 - 180
        cam_az += diff * (1 - cam_decay ** steps)
        viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], data.qpos[2] + 0.3]
        viewer.cam.azimuth = cam_az

        viewer.sync()

        # 실시간 배속: 1.00 보다 한참 낮으면 화면이 물리를 못 따라오는 것이다
        wall = time.time() - start
        if wall - last_print > 0.5:
            print(f"\r전진 {cmd['v']:+.2f} m/s  회전 {cmd['w']:+.2f}  "
                  f"실시간 {data.time/wall:4.2f}배 ", end="", flush=True)
            last_print = wall
        wait = data.time - wall
        if wait > 0:
            time.sleep(wait)
