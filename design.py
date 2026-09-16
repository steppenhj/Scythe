"""설계 검토용 3D 장면 — 차체와 절삭 헤드 후보를 나란히 놓고 본다.

지형은 안 띄운다. 여기서 보려는 건 '땅 위에서 어떻게 도나'가 아니라
'이 물건이 대체 어떤 크기이고 풀에 비해 어떤 꼴인가' 다. 평지에 세워놓고
눈금자를 옆에 꽂는 쪽이 훨씬 빠르고 잘 보인다.

네 대가 왼쪽부터 늘어서 있다. 차체는 넷 다 같다 — 다른 건 머리뿐이다.

  0  머리 없음 (지금 robot.py 상태)
  1  플레일 드럼      누운 드럼이 돌며 풀을 위에서 때려 부순다. 날 중심 0.19 m
  2  나일론 줄 헤드    수직축 원반이 정해진 높이에서 싹둑 자른다. 날 0.13 m
  3  2단 절삭         앞 드럼이 0.52 m 에서 윗동을 치고, 차체 밑 데크가 마무리

같이 세워둔 것들이 이 장면의 핵심이다:
  · 빨강/흰색 눈금 기둥 — 한 칸 0.1 m, 1.0 m 까지
  · 반투명 초록 덩어리 — 벌초 전 풀 (0.9 m, 허리 높이)
  · 회색 사람 — 키 1.7 m
  · 봉분과 상석 — 지름 2.2 m, 높이 0.8 m / 1.2×0.7×0.25 m

조종:
  G  풀 껐다 켜기 (안이 안 보일 때)
  1 2 3 0  해당 로봇으로 카메라 이동   Q 전체 보기
  마우스 왼쪽 드래그 회전, 휠 확대

실행:  .venv/bin/python design.py
"""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import time

import mujoco
import mujoco.viewer

# ────── 차체 (robot.py 와 같은 값) ──────
BODY_L, BODY_W, BODY_H = 0.80, 0.50, 0.20
TRACK_W, TRACK_H = 0.15, 0.12
BODY_M, TRACK_M = 25.0, 5.0

# ────── 풀·사람 ──────
GRASS_H = 0.90        # 벌초 전 풀 높이 (허리) — 사용자 확인값
PERSON_H = 1.70

# ────── 절삭 헤드 후보 [전부 추정 — 카탈로그 보고 맞출 값] ──────
# kind 가 모양을 정한다. 이게 셋의 진짜 차이다:
#   drum — 좌우로 누운 드럼이 돌며 위에서 아래로 때린다 (플레일)
#   disc — 수직축으로 도는 납작한 원반. 정해진 높이에서 싹둑 자른다 (나일론 줄)
# (이름, kind, 앞으로 내민 거리, 폭/지름, 두께(drum만), 날 중심 높이, 질량, 색)
HEADS = [
    ("머리 없음",   None,   0.00, 0.00, 0.00, 0.00,  0.0, ".8 .3 .2 1"),
    ("플레일 드럼", "drum", 0.62, 0.60, 0.34, 0.19, 26.0, ".35 .35 .40 1"),
    ("나일론 줄",   "disc", 0.55, 0.55, 0.00, 0.13,  6.0, ".85 .75 .20 1"),
    ("2단 절삭",    "drum", 0.60, 0.60, 0.22, 0.52, 14.0, ".30 .55 .75 1"),
]
SPACING = 3.2         # 로봇 사이 간격
SHADOW = 1024

_g = []               # 풀 geom 이름 — G 키로 토글


def _pole(x, y, n=10):
    """눈금 기둥: 0.1 m 짜리 칸을 빨강·흰색으로 번갈아 쌓는다."""
    out = []
    for i in range(n):
        c = ".85 .15 .15 1" if i % 2 == 0 else ".95 .95 .95 1"
        out.append(f'<geom type="cylinder" size=".025 .05" rgba="{c}" '
                   f'contype="0" conaffinity="0" pos="{x:.2f} {y:.2f} {i*0.1+0.05:.2f}"/>')
    return "\n    ".join(out)


def _person(x, y):
    """키 1.7 m 사람 — 크기 감각용."""
    return "\n    ".join([
        f'<geom type="cylinder" size=".16 .40" rgba=".45 .45 .50 1" contype="0" '
        f'conaffinity="0" pos="{x:.2f} {y:.2f} .40"/>',
        f'<geom type="cylinder" size=".21 .32" rgba=".55 .55 .60 1" contype="0" '
        f'conaffinity="0" pos="{x:.2f} {y:.2f} 1.12"/>',
        f'<geom type="sphere" size=".115" rgba=".65 .60 .55 1" contype="0" '
        f'conaffinity="0" pos="{x:.2f} {y:.2f} 1.56"/>',
    ])


def _robot(i, name, kind, reach, hw, hd, hz, hm, color):
    """차체 + 머리. x = i*SPACING 에 세운다."""
    x = i * SPACING
    parts = [
        f'<geom name="chassis{i}" type="box" '
        f'size="{BODY_L/2} {BODY_W/2} {BODY_H/2}" rgba="{color}" '
        f'pos="{x:.2f} 0 {TRACK_H/2 + BODY_H/2:.3f}"/>',
        f'<geom name="ltrack{i}" type="box" '
        f'size="{BODY_L/2} {TRACK_W/2} {TRACK_H/2}" rgba=".15 .15 .15 1" '
        f'pos="{x:.2f} {(BODY_W+TRACK_W)/2:.3f} 0"/>',
        f'<geom name="rtrack{i}" type="box" '
        f'size="{BODY_L/2} {TRACK_W/2} {TRACK_H/2}" rgba=".15 .15 .15 1" '
        f'pos="{x:.2f} {-(BODY_W+TRACK_W)/2:.3f} 0"/>',
    ]
    if kind == "drum":
        # 좌우로 누운 드럼 → 원기둥 축을 y 로 눕힌다 (zaxis="0 1 0")
        parts.append(
            f'<geom name="head{i}" type="cylinder" size="{hd/2:.3f} {hw/2:.3f}" '
            f'zaxis="0 1 0" rgba="{color}" pos="{x + reach:.3f} 0 {hz:.3f}"/>')
    elif kind == "disc":
        # 수직축 원반. 얇다 — 두께가 아니라 지름이 절삭폭이다
        parts.append(
            f'<geom name="head{i}" type="cylinder" size="{hw/2:.3f} .012" '
            f'rgba="{color}" pos="{x + reach:.3f} 0 {hz:.3f}"/>')
    if kind:        # 머리를 차체에 잇는 팔 두 개
        for s in (1, -1):
            parts.append(
                f'<geom type="box" size="{reach/2:.3f} .03 .03" rgba=".3 .3 .3 1" '
                f'pos="{x + reach/2:.3f} {s*hw/2*0.75:.3f} {hz:.3f}"/>')
    if i == 3:      # 2단: 앞이 윗동을 치고, 차체 밑 이 데크가 마무리한다
        parts.append(
            f'<geom type="cylinder" size=".26 .012" rgba=".30 .55 .75 1" '
            f'pos="{x:.2f} 0 .075"/>')
    return "\n    ".join(parts)


def _grass(i):
    """로봇을 감싸는 풀 덩어리. 반투명이라 안이 비친다."""
    x = i * SPACING
    _g.append(f"grass{i}")
    return (f'<geom name="grass{i}" type="box" size="1.30 1.10 {GRASS_H/2:.3f}" '
            f'rgba=".22 .45 .18 .30" contype="0" conaffinity="0" '
            f'pos="{x:.2f} 0 {GRASS_H/2:.3f}"/>')


SCENE = "\n    ".join(
    [_robot(i, *h) for i, h in enumerate(HEADS)]
    + [_grass(i) for i in range(len(HEADS))]
    + [_pole(i * SPACING - 1.45, -1.0) for i in range(len(HEADS))]
    + [_person(-2.2, 0.0),
       # 봉분과 상석 — 넘을 수 없다는 걸 눈으로 확인하는 용도
       f'<geom type="ellipsoid" size="1.10 1.10 0.80" rgba=".38 .46 .30 1" '
       f'pos="{len(HEADS)*SPACING - 0.4:.2f} 3.0 0"/>',
       f'<geom type="box" size=".60 .35 .125" rgba=".6 .6 .62 1" '
       f'pos="{len(HEADS)*SPACING - 0.4:.2f} 1.3 .125"/>'])

XML = f"""
<mujoco>
  <option gravity="0 0 -9.81"/>
  <visual>
    <quality shadowsize="{SHADOW}" offsamples="0" numslices="16" numstacks="10"/>
  </visual>
  <asset>
    <texture name="chk" type="2d" builtin="checker"
             rgb1=".30 .34 .26" rgb2=".36 .40 .30" width="300" height="300"/>
    <material name="ground" texture="chk" texuniform="true" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light directional="true" pos="-4 -6 8" dir=".3 .5 -1" diffuse=".95 .95 .95"/>
    <geom type="plane" size="30 30 .1" material="ground"/>
    {SCENE}
  </worldbody>
</mujoco>
"""


def report():
    """무게중심과 앞으로 고꾸라지는 각도. 머리를 앞에 달면 여기가 움직인다."""
    base_m = BODY_M + 2 * TRACK_M
    # 차체만의 무게중심 높이 (궤도 중심을 0 으로)
    base_z = (BODY_M * (TRACK_H / 2 + BODY_H / 2) + 2 * TRACK_M * 0) / base_m
    print(f"{'':14} {'총무게':>7} {'무게중심 앞으로':>14} {'무게중심 높이':>12} "
          f"{'앞으로 고꾸라지는 각':>18}")
    for name, _k, reach, hw, hd, hz, hm, _c in HEADS:
        m = base_m + hm
        cx = hm * reach / m
        cz = (base_m * base_z + hm * hz) / m
        # 앞 궤도 끝(+BODY_L/2)을 넘어가면 넘어간다
        import math
        ang = math.degrees(math.atan2(BODY_L / 2 - cx, cz))
        print(f"  {name:12} {m:6.1f} kg {cx:12.3f} m {cz:11.3f} m {ang:15.1f}°")
    print(f"\n  (궤도 접지 길이 {BODY_L} m, 지금 땅 기울기 4.6° — 여유가 얼마나 남나 보는 값)")
    print("  헤드 무게·치수는 전부 [추정]. 카탈로그 보고 맞출 것.")


if __name__ == "__main__":
    print(__doc__)
    report()

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    GIDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in _g]
    view = {"target": None}

    def key_callback(key):
        if key == ord("G"):
            for g in GIDS:                       # 알파만 0 과 0.30 사이에서 토글
                model.geom_rgba[g][3] = 0.30 if model.geom_rgba[g][3] < 0.01 else 0.0
        elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
            view["target"] = int(chr(key))
        elif key == ord("Q"):
            view["target"] = None

    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=key_callback) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation = 128, -18
        viewer.cam.lookat[:] = [len(HEADS) * SPACING / 2 - 1, 0, 0.5]
        viewer.cam.distance = 13.0
        while viewer.is_running():
            t = view["target"]
            if t is None:
                viewer.cam.lookat[:] = [len(HEADS) * SPACING / 2 - 1, 0, 0.5]
                viewer.cam.distance = 13.0
            else:
                viewer.cam.lookat[:] = [t * SPACING, 0, 0.45]
                viewer.cam.distance = 3.4
            viewer.sync()
            time.sleep(1 / 20)      # 움직이는 게 없다 — 성기게 그린다
