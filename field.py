"""Phase 1 입구: 산소 지형 위에서 로봇을 몬다.

지형 좌표계: 원점은 전체 땅 ∪ 예초 구역의 도심, +x 동쪽, +y 북쪽 (m).
내리막은 동남동(방위각 120°), 4.6도.

지형은 terrain.py 에서 재료로 가져오고(뷰어는 안 뜬다), 차체와 조종은
robot.py 와 같다. robot.py(평지)가 대조군, 이 파일(지형)이 실험군 —
둘의 차이는 오직 '땅'뿐이어야 비교가 성립한다.

조종 (뷰어 창을 클릭한 뒤):
  ↑/W 속도 한 칸 올림   ↓/S 한 칸 내림   (크루즈처럼 유지)
  ←/A →/D 조향 — 놓으면 직진 복귀   스페이스/X 정지
  R 출발 지점으로 리셋 (뒤집히거나 둔덕 아래로 떨어졌을 때)

실행:  .venv/bin/python field.py
"""
import math
import os
import time

# WSLg + Mesa 21.2: 하드웨어 GL이 조용히 실패한다 → CPU 렌더링
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import mujoco
import mujoco.viewer

import terrain   # 산소 지형 재료: XML 조각들, 높이 함수 surf(), 격자 Z
                # (지형 숫자의 출처는 terrain_data.py ← trace.py ← photos/)

# ────── 차체: robot.py 와 같은 가설 (A안 35 kg) ──────
BODY_L = 0.80
BODY_W = 0.50
BODY_H = 0.20
TRACK_W = 0.15
TRACK_H = 0.12
BODY_M, TRACK_M = 25.0, 5.0
FRICTION = 0.6
MAX_V = 1.0
STEP_V = 0.25
STEER_V = 0.5
STEER_TAU = 0.6
RENDER_HZ = 50    # 화면 갱신 (Hz). WSLg CPU 렌더링이 버거우면 30, 20 으로 내린다.
                  # 물리는 이것과 무관하게 2 ms 마다 푼다 — 화면만 성기어진다
START = (12.0, 1.8)    # 예초 구역 안, 경계에서 7.7 m·봉분에서 8.0 m 떨어진 트인 자리
# ──────────────────────────────────────────────────────

START_Z = terrain.surf(*START) + TRACK_H / 2 + 0.05

XML = f"""
<mujoco>
  <option gravity="0 0 -9.81" cone="elliptic"/>

  <asset>
    <hfield name="myoyeok" nrow="{terrain.NROW}" ncol="{terrain.NCOL}"
            size="{terrain.RX} {terrain.RY} {terrain.ZMAX:.4f} 0.5"/>
    <texture name="grid" type="2d" builtin="checker"
             rgb1=".28 .36 .22" rgb2=".34 .42 .26" width="300" height="300"/>
    <material name="grass" texture="grid" texuniform="true" texrepeat="2 2"/>
  </asset>

  <worldbody>
    <light directional="true" pos="0 0 20" dir="-.2 .2 -1" diffuse=".9 .9 .9"/>

    <geom type="hfield" hfield="myoyeok" material="grass"
          friction="{FRICTION} .005 .0001"/>

    {terrain.SANGSEOK}
    {terrain.TREE_GEOMS}
    {terrain.STAKES}

    <body name="robot" pos="{START[0]} {START[1]} {START_Z:.3f}">
      <freejoint/>
      <geom name="chassis" type="box"
            size="{BODY_L / 2} {BODY_W / 2} {BODY_H / 2}"
            pos="0 0 {TRACK_H / 2 + BODY_H / 2:.3f}"
            mass="{BODY_M}" rgba=".8 .3 .2 1"/>
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
      <light mode="trackcom" pos="0 0 4" dir="0 0 -1" diffuse=".8 .8 .8"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
model.hfield_data[:] = ((terrain.Z - terrain.ZMIN) / terrain.ZMAX).ravel()
data = mujoco.MjData(model)

LT = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ltrack")
RT = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rtrack")
assert model.flg_surfacevel, "surfacevel 플래그가 꺼져 있다 (XML 선언 확인)"

cmd = {"v": 0.0, "w": 0.0}
flags = {"reset": False}

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
    elif key == ord("R"):                 # 리셋 (실제 처리는 메인 루프에서)
        flags["reset"] = True
    cmd["v"] = max(-MAX_V, min(MAX_V, cmd["v"]))   # 상태는 메인 루프가 찍는다

def apply_tracks():
    """궤도 표면 속도 쓰기. 표면이 뒤(-x)로 돌아야 차체가 앞(+x)으로 간다."""
    vl = cmd["v"] - cmd["w"]
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
    viewer.cam.distance = 5.0
    viewer.cam.elevation = -20

    steer_decay = math.exp(-model.opt.timestep / STEER_TAU)
    cam_decay = math.exp(-model.opt.timestep / 0.7)
    cam_az = yaw_of(data.qpos[3:7])
    steps = max(1, round(1 / RENDER_HZ / model.opt.timestep))

    start = time.time()
    last_print = 0.0
    while viewer.is_running():
        if flags["reset"]:
            mujoco.mj_resetData(model, data)   # XML 초기 상태(출발 지점)로
            cmd["v"] = cmd["w"] = 0.0
            flags["reset"] = False
            start = time.time()                # 시계도 같이 되감는다

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
