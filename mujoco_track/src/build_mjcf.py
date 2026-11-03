# -*- coding: utf-8 -*-
"""subject3_scaled.osim + newmus_paths_k8.json → MJCF (하지 4자유도×2 + 케이블 16, 골반 자유).

사용: python3 -I build_mjcf.py <osim> <paths.json> <walking_ik.mot> <out.xml>
"""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np
from osim_model import OsimModel, mat2quat, read_mot, mot_to_qdicts, axis_angle

osim_path, json_path, ik_path, out_path = sys.argv[1:5]
DYN = len(sys.argv) > 5 and sys.argv[5] == '--dyn'   # 순동역학용: 잠금 관절을 평균값으로 굳혀 바디에 흡수 (등식 구속 없음)
M = OsimModel(osim_path)
P = json.load(open(json_path))
hdr, data = read_mot(ik_path)
qs = mot_to_qdicts(hdr, data)

LEG_CHAIN = ['femur', 'tibia', 'talus', 'calcn', 'toes']
# 코딩된 자유도 (MuJoCo 관절명 = OpenSim 좌표명)
LOCK_CONST = ['hip_rotation', 'subtalar_angle', 'mtp_angle']


def fmt(v):
    return ' '.join(f'{x:.8g}' for x in np.asarray(v).ravel())


BAKE_PREFIX = ('hip_rotation', 'subtalar', 'mtp', 'lumbar', 'arm_', 'elbow', 'pro_sup', 'wrist')


def is_baked(coord):
    return DYN and any(coord.startswith(p) for p in BAKE_PREFIX)


def body_pose(J, qfix=None):
    """child body frame in parent body frame — 굳힌 좌표는 qfix 값, 나머지 0."""
    q = {c: (qfix.get(c, 0.0) if qfix else 0.0) for c in J.coords}
    return M.joint_xform(J, q)


def knee_poly(J, deg=4, qmax=2.1):
    """walker_knee 의 종속 축(rotation2, rotation3, translation1, translation2)을 knee_angle 의 다항식으로 적합."""
    x = np.linspace(0, qmax, 200)
    out = {}
    for ax in J.axes:
        if ax.coord is None or ax.label == 'rotation1':
            continue
        y = np.array([ax.func(v) for v in x])
        c = np.polyfit(x, y, deg)[::-1]  # c0..c4
        yy = np.polyval(c[::-1], x)
        out[ax.label] = (c, float(np.max(np.abs(yy - y))), float(np.max(np.abs(y))))
    return out


lines = []
A = lines.append
A('<mujoco model="panki_leg16_subject3">')
A('  <compiler angle="radian" inertiafromgeom="false" autolimits="true"/>')
A('  <option timestep="0.0005" gravity="0 -9.81 0" integrator="implicitfast" cone="elliptic" impratio="10"/>')
A('  <size memory="20M"/>')
A('  <visual><global offwidth="1280" offheight="720"/><headlight ambient="0.4 0.4 0.4"/></visual>')
A('  <default>')
A('    <geom contype="0" conaffinity="0" rgba="0.75 0.72 0.7 1"/>')
A('    <default class="foot"><geom type="sphere" size="0.018" contype="1" conaffinity="1" friction="1.0 0.005 0.0001" solref="0.015 1" solimp="0.9 0.95 0.001" rgba="0.3 0.3 0.35 1"/></default>')
A('    <joint damping="0.5" armature="0.001" limited="true" solreflimit="0.01 1" solimplimit="0.9 0.95 0.001"/>')
A('    <site size="0.006" rgba="0.8 0.1 0.2 1"/>')
A('    <tendon width="0.004" rgba="0.6 0.1 0.15 1"/>')
A('    <default class="sma"><general ctrllimited="true" ctrlrange="0 1" gaintype="fixed" biastype="none" dyntype="none"/></default>')
A('  </default>')
A('  <worldbody>')
A('    <light pos="0 3 2" dir="0 -1 -0.5" directional="true"/>')
A('    <geom name="floor" type="plane" size="20 20 0.1" quat="0.70710678 -0.70710678 0 0" contype="1" conaffinity="1" friction="1.0 0.005 0.0001" rgba="0.9 0.9 0.9 1"/>')

polyinfo = {}
equalities = []
joint_order = []   # MuJoCo joint names in order (for qpos mapping)


def emit_body(name, parent_joint, indent, extra_children=None, locked_q=None):
    """재귀로 바디 XML 생성. parent_joint: 이 바디를 부모에 붙이는 OpenSim 관절."""
    b = M.bodies[name]
    J = parent_joint
    pad = ' ' * indent
    qfix = {c: locked_q[c] for c in J.coords if locked_q and c in locked_q and is_baked(c)}
    R, t = body_pose(J, qfix)
    quat = mat2quat(R)
    A(f'{pad}<body name="{name}" pos="{fmt(t)}" quat="{fmt(quat)}">')
    I = b.inertia
    A(f'{pad}  <inertial pos="{fmt(b.com)}" mass="{b.mass:.6g}" fullinertia="{I[0]:.8g} {I[1]:.8g} {I[2]:.8g} {I[3]:.8g} {I[4]:.8g} {I[5]:.8g}"/>')
    # joints (child frame 기준 축·앵커)
    side = name.split('_')[-1]
    base = name.rsplit('_', 1)[0] if side in ('r', 'l') else name
    if J.kind == 'CustomJoint' and J.name.startswith('walker_knee'):
        # slides first (translation1: y, translation2: z), then rotations (1: x primary, 2: z, 3: y)
        kp = knee_poly(J)
        polyinfo[J.name] = kp
        for ax in J.axes:
            if ax.kind == 'trans' and ax.coord:
                jn = f'{J.name}_{ax.label}'
                axis_c = J.R_c @ ax.axis
                A(f'{pad}  <joint name="{jn}" type="slide" pos="{fmt(J.t_c)}" axis="{fmt(axis_c)}" range="-0.02 0.02" damping="0"/>')
                joint_order.append(jn)
                c = kp[ax.label][0]
                equalities.append((jn, J.coords[0], c))
        for ax in J.axes:
            if ax.kind == 'rot':
                axis_c = J.R_c @ ax.axis
                if ax.label == 'rotation1':
                    jn = J.coords[0]
                    A(f'{pad}  <joint name="{jn}" type="hinge" pos="{fmt(J.t_c)}" axis="{fmt(axis_c)}" range="-0.05 2.27"/>')
                    joint_order.append(jn)
                else:
                    jn = f'{J.name}_{ax.label}'
                    A(f'{pad}  <joint name="{jn}" type="hinge" pos="{fmt(J.t_c)}" axis="{fmt(axis_c)}" range="-0.5 0.5" damping="0"/>')
                    joint_order.append(jn)
                    c = kp[ax.label][0]
                    equalities.append((jn, J.coords[0], c))
    elif J.kind in ('CustomJoint', 'PinJoint', 'UniversalJoint'):
        for ax in J.axes:
            if ax.kind == 'rot' and ax.coord:
                if is_baked(ax.coord):
                    continue
                slope = ax.func(1.0) - ax.func(0.0)   # OpenSim 은 왼쪽 좌표를 축 반전이 아니라 LinearFunction(-1)로 뒤집는다
                assert abs(abs(slope) - 1) < 1e-9 and abs(ax.func(0.0)) < 1e-9, (J.name, ax.label, slope)
                axis_c = J.R_c @ (ax.axis * slope)
                jn = ax.coord
                ROM = {'hip_flexion': (-0.52, 2.1), 'hip_adduction': (-0.79, 0.52), 'ankle_angle': (-0.87, 0.52)}
                key = jn.rsplit('_', 1)[0]
                rng = f'{ROM[key][0]} {ROM[key][1]}' if key in ROM else '-3 3'
                A(f'{pad}  <joint name="{jn}" type="hinge" pos="{fmt(J.t_c)}" axis="{fmt(axis_c)}" range="{rng}"/>')
                joint_order.append(jn)
                if locked_q is not None and jn in locked_q:
                    equalities.append((jn, None, [locked_q[jn], 0, 0, 0, 0]))
    # visual capsule to each child joint center
    for Jc in M.joints:
        if Jc.parent == name and Jc.child in M.bodies:
            p = Jc.t_p
            L = np.linalg.norm(p)
            if L > 0.03:
                A(f'{pad}  <geom type="capsule" fromto="0 0 0 {fmt(p)}" size="0.02"/>')
    if name.startswith('calcn'):
        # 발바닥 접촉 구 4개 (뒤꿈치·중족 외/내·엄지 아래) — 위치는 빌드 후 sole_y 로 보정
        A(f'{pad}  <geom class="foot" name="{name}_heel" pos="0.005 SOLE 0.0"/>')
        A(f'{pad}  <geom class="foot" name="{name}_mid_l" pos="0.09 SOLE 0.03"/>')
        A(f'{pad}  <geom class="foot" name="{name}_mid_m" pos="0.09 SOLE -0.03"/>')
        A(f'{pad}  <geom class="foot" name="{name}_toe" pos="0.165 SOLE 0.0"/>')
        A(f'{pad}  <geom type="box" pos="0.085 -0.02 0" size="0.09 0.012 0.035" rgba="0.5 0.45 0.4 1"/>')
    if name.startswith('toes'):
        A(f'{pad}  <geom class="foot" name="{name}_tip" pos="0.045 SOLE_T 0.0"/>')
    # cable sites
    for m in P['muscles']:
        for s in ('r', 'l'):
            for k, (bname, pt) in enumerate(zip(m['chain'], m['points'])):
                bn = bname if bname == 'pelvis' else bname[:-2] + '_' + s
                if bn == name:
                    pt = np.array(pt, float)
                    if s == 'l':
                        pt = pt * np.array([1, 1, -1])
                    A(f'{pad}  <site name="{m["name"]}_{s}_{k}" pos="{fmt(pt)}"/>')
    # children
    for Jc in M.joints:
        if Jc.parent == name and Jc.child in M.bodies:
            ch = Jc.child
            if ch.startswith('patella'):
                continue
            emit_body(ch, Jc, indent + 2, locked_q=locked_q)
    A(f'{pad}</body>')


# 잠금 관절의 기준값 (walking1 평균) — hip_rotation·subtalar·mtp·lumbar·arm
mean_q = {k: float(np.mean([q[k] for q in qs])) for k in hdr[1:]}
locked = {}
for c in hdr[1:]:
    if any(c.startswith(p) for p in ('hip_rotation', 'subtalar', 'mtp', 'lumbar', 'arm_', 'elbow', 'pro_sup', 'wrist')):
        locked[c] = mean_q[c]
# 골반
gp = M.joint_by_child['pelvis']
A('    <body name="pelvis" pos="0 0.95 0">')
A('      <freejoint name="pelvis"/>')
joint_order.append('pelvis')
b = M.bodies['pelvis']
I = b.inertia
A(f'      <inertial pos="{fmt(b.com)}" mass="{b.mass:.6g}" fullinertia="{I[0]:.8g} {I[1]:.8g} {I[2]:.8g} {I[3]:.8g} {I[4]:.8g} {I[5]:.8g}"/>')
A('      <geom type="box" pos="-0.07 0 0" size="0.06 0.07 0.12"/>')
A('      <site name="pelvis_c" pos="0 0 0" size="0.01"/>')
for m in P['muscles']:
    for s in ('r', 'l'):
        for k, (bname, pt) in enumerate(zip(m['chain'], m['points'])):
            if bname == 'pelvis':
                pt = np.array(pt, float)
                if s == 'l':
                    pt = pt * np.array([1, 1, -1])
                A(f'      <site name="{m["name"]}_{s}_{k}" pos="{fmt(pt)}"/>')
for Jc in M.joints:
    if Jc.parent == 'pelvis' and Jc.child in M.bodies:
        emit_body(Jc.child, Jc, 6, locked_q=locked)
A('    </body>')
A('  </worldbody>')
# equality: 무릎 종속 축 (다항식) + 잠금 관절
A('  <equality>')
for jn, j2, c in equalities:
    if j2 is None:
        A(f'    <joint name="eq_{jn}" joint1="{jn}" polycoef="{fmt(c)}" solref="0.002 1"/>')
    else:
        A(f'    <joint name="eq_{jn}" joint1="{jn}" joint2="{j2}" polycoef="{fmt(c)}" solref="0.002 1"/>')
A('  </equality>')
# tendons + actuators
A('  <tendon>')
for m in P['muscles']:
    for s in ('r', 'l'):
        A(f'    <spatial name="{m["name"]}_{s}">')
        for k in range(len(m['chain'])):
            A(f'      <site site="{m["name"]}_{s}_{k}"/>')
        A('    </spatial>')
A('  </tendon>')
A('  <actuator>')
for m in P['muscles']:
    for s in ('r', 'l'):
        A(f'    <general class="sma" name="{m["name"]}_{s}" tendon="{m["name"]}_{s}" gainprm="{-m["force_N"]:.6g}"/>')
A('  </actuator>')
A('</mujoco>')
xml = '\n'.join(lines)

# ---- sole height from FK on walking frames (flat foot): calcn/toes candidate points
def sole_height():
    hmin = {'calcn': 1e9, 'toes': 1e9}
    for q in qs[::3]:
        fk = M.fk(q)
        for s in ('r', 'l'):
            for pt in ([0.005, 0, 0], [0.09, 0, 0.03], [0.09, 0, -0.03], [0.165, 0, 0]):
                y = M.point_world(fk, f'calcn_{s}', pt)[1]
                hmin['calcn'] = min(hmin['calcn'], y)
            y = M.point_world(fk, f'toes_{s}', [0.045, 0, 0])[1]
            hmin['toes'] = min(hmin['toes'], y)
    return hmin

h = sole_height()
r = 0.018
sole = -(h['calcn'] - r)   # 구 중심 y (local) : 바닥에서 반지름만큼 위
sole_t = -(h['toes'] - r)
xml = xml.replace('SOLE_T', f'{sole_t:.4f}').replace('SOLE', f'{sole:.4f}')
open(out_path, 'w').write(xml)
json.dump({'joint_order': joint_order, 'locked': locked, 'mean_q': mean_q,
           'knee_poly': {k: {lab: {'coef': list(map(float, v[0])), 'maxerr': v[1], 'range': v[2]} for lab, v in kp.items()} for k, kp in polyinfo.items()},
           'sole_local_y': sole, 'sole_toe_local_y': sole_t, 'foot_min_world_y': h},
          open(out_path.replace('.xml', '_meta.json'), 'w'), indent=1)
print('wrote', out_path, 'joints', len(joint_order), 'equalities', len(equalities))
print('foot min world y (walking1):', h, '→ sole local y', round(sole, 4), round(sole_t, 4))
for k, kp in polyinfo.items():
    for lab, v in kp.items():
        print(f'  {k} {lab}: poly4 maxerr {v[1]*1000:.3f} (mm or mrad) of range {v[2]*1000:.1f}')
