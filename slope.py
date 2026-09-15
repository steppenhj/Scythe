"""경사면에 상자를 올려놓고 미끄러지는지 본다.

실행:  .venv/bin/python slope.py
마우스 왼쪽 드래그로 시점 회전, 휠로 확대.
"""
import math
import os
import time

# WSLg의 d3d12 GPU 드라이버를 Ubuntu 20.04의 Mesa 21.2가 못 쓴다.
# 그대로 두면 창은 뜨는데 화면이 검게만 나온다. CPU 렌더링으로 우회.
# (mujoco import 전에 설정해야 한다)
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import mujoco
import mujoco.viewer

# ────────── 이 두 숫자만 바꿔가며 실험한다 ──────────
SLOPE_DEG = 31.0   # 경사각 (도)
FRICTION = 0.6     # 마찰계수 (0=얼음, 1=고무)
# ──────────────────────────────────────────────────

XML = f"""
<mujoco>
  <option gravity="0 0 -9.81"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1=".2 .3 .4" rgb2=".3 .4 .5" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="160 160"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>

    <!-- 경사면: euler의 두 번째 값이 Y축 기준 기울기(도).
         size는 '그려주는 범위'일 뿐, 충돌 계산상 plane은 무한히 넓다 -->
    <geom name="slope" type="plane" size="100 100 .1"
          euler="0 {SLOPE_DEG} 0"
          friction="{FRICTION} .005 .0001" material="grid"/>

    <!-- 상자: freejoint = 6자유도로 자유롭게 움직인다 -->
    <body name="box" pos="0 0 .3">
      <freejoint/>
      <geom type="box" size=".15 .15 .15" mass="5"
            friction="{FRICTION} .005 .0001" rgba=".8 .3 .2 1"/>
      <!-- 상자를 따라다니는 카메라. 없으면 상자가 화면 밖으로 사라진다 -->
      <camera name="track" mode="trackcom"
              pos="0 -4 2" xyaxes="1 0 0  0 0.447 0.894"/>
      <!-- 조명도 같이 따라가야 한다. 고정이면 멀어질수록 어두워진다 -->
      <light mode="trackcom" pos="0 0 4" dir="0 0 -1" diffuse=".8 .8 .8"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)   # 안 변하는 것: 질량, 마찰, 모양
data = mujoco.MjData(model)                   # 매 스텝 변하는 것: 위치, 속도

print(f"경사 {SLOPE_DEG}도, 마찰 {FRICTION}")
print(f"이론상 미끄러지기 시작하는 각도: {math.degrees(math.atan(FRICTION)):.1f}도")
print("창을 닫으면 끝납니다.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 위에서 만든 track 카메라를 시작 시점부터 쓴다.
    # (그냥 두면 원점 고정 카메라라, 상자가 나가면 빈 화면만 남는다)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "track")

    start = time.time()
    while viewer.is_running():
        mujoco.mj_step(model, data)   # 물리 한 스텝 전진 (기본 2ms)
        viewer.sync()                 # 화면 갱신

        # 실시간 속도로 맞춰 재생
        wait = data.time - (time.time() - start)
        if wait > 0:
            time.sleep(wait)

print(f"\n{data.time:.1f}초 동안 상자가 미끄러진 거리: {abs(data.qpos[0]):.2f} m")
