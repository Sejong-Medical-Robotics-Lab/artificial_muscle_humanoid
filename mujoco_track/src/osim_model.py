# -*- coding: utf-8 -*-
"""OpenSim(.osim, Rajagopal 계열) → 해석용 구조체 + numpy 순기구학 + MuJoCo MJCF 생성.

독립 구현 원칙: MJCF 생성과 numpy FK는 같은 파싱 결과를 쓰지만, 기구학 계산은 서로 다른 코드
(MuJoCo 엔진 vs numpy)이므로 사이트 위치·텐던 길이 일치가 변환의 정확성 검증이 된다.
좌표계: OpenSim 그대로(Y 위, X 앞, Z 오른쪽). MuJoCo 중력은 (0, -9.81, 0).
"""
from __future__ import annotations
import json, math
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_xyz(v):
    """OpenSim 'orientation' = body-fixed X-Y-Z 순서: R = Rx·Ry·Rz."""
    return rot_x(v[0]) @ rot_y(v[1]) @ rot_z(v[2])


def axis_angle(axis, q):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


def mat2quat(R):
    """회전행렬 → (w,x,y,z)."""
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


@dataclass
class Axis:
    kind: str            # 'rot' | 'trans'
    axis: np.ndarray     # joint-frame axis
    coord: str | None    # coordinate name (None → constant)
    func: object         # callable(q)->value  (q in rad / m)
    label: str = ''


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    t_p: np.ndarray
    R_p: np.ndarray
    t_c: np.ndarray
    R_c: np.ndarray
    axes: list = field(default_factory=list)   # ordered rot1..3 then trans1..3 (as in osim)
    coords: list = field(default_factory=list)


@dataclass
class Body:
    name: str
    mass: float
    com: np.ndarray
    inertia: np.ndarray  # xx yy zz xy xz yz


def _vec(txt):
    return np.array([float(x) for x in txt.split()])


def _parse_function(el):
    """TransformAxis 안의 함수 요소 → callable."""
    tag = el.tag
    if tag == 'LinearFunction':
        c = _vec(el.findtext('coefficients'))
        return lambda q, c=c: c[0] * q + c[1]
    if tag == 'Constant':
        v = float(el.findtext('value'))
        return lambda q, v=v: v
    if tag in ('SimmSpline', 'GCVSpline', 'NaturalCubicSpline'):
        x = _vec(el.findtext('x'))
        y = _vec(el.findtext('y'))
        cs = CubicSpline(x, y, bc_type='natural')
        return lambda q, cs=cs: float(cs(q))
    if tag == 'MultiplierFunction':
        inner = _parse_function(el.find('function')[0])
        sc = float(el.findtext('scale'))
        return lambda q, inner=inner, sc=sc: sc * inner(q)
    raise ValueError('unknown function ' + tag)


class OsimModel:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        model = root.find('Model')
        self.name = model.get('name')
        self.bodies = {}
        for b in model.find('BodySet/objects'):
            self.bodies[b.get('name')] = Body(b.get('name'), float(b.findtext('mass')), _vec(b.findtext('mass_center')), _vec(b.findtext('inertia')))
        self.joints = []
        self.coord_default = {}
        for j in model.find('JointSet/objects'):
            frames = {f.get('name'): f for f in j.find('frames')}
            pf = frames[j.findtext('socket_parent_frame').strip()]
            cf = frames[j.findtext('socket_child_frame').strip()]
            parent = pf.findtext('socket_parent').strip().split('/')[-1]
            child = cf.findtext('socket_parent').strip().split('/')[-1]
            J = Joint(j.get('name'), j.tag, parent, child, _vec(pf.findtext('translation')), euler_xyz(_vec(pf.findtext('orientation'))),
                      _vec(cf.findtext('translation')), euler_xyz(_vec(cf.findtext('orientation'))))
            J.coords = [c.get('name') for c in j.findall('coordinates/Coordinate')]
            for c in j.findall('coordinates/Coordinate'):
                self.coord_default[c.get('name')] = float(c.findtext('default_value') or 0.0)
            if j.tag == 'CustomJoint':
                st = j.find('SpatialTransform')
                for ta in st.findall('TransformAxis'):
                    nm = ta.get('name')
                    coords = (ta.findtext('coordinates') or '').split()
                    fn_el = None
                    for ch in ta:
                        if ch.tag not in ('coordinates', 'axis'):
                            fn_el = ch
                    func = _parse_function(fn_el) if fn_el is not None else (lambda q: 0.0)
                    J.axes.append(Axis('rot' if nm.startswith('rotation') else 'trans', _vec(ta.findtext('axis')), coords[0] if coords else None, func, nm))
            elif j.tag == 'PinJoint':
                J.axes.append(Axis('rot', np.array([0, 0, 1.0]), J.coords[0], lambda q: q, 'rotation1'))
            elif j.tag == 'UniversalJoint':
                J.axes.append(Axis('rot', np.array([1.0, 0, 0]), J.coords[0], lambda q: q, 'rotation1'))
                J.axes.append(Axis('rot', np.array([0, 1.0, 0]), J.coords[1], lambda q: q, 'rotation2'))
            elif j.tag == 'WeldJoint':
                pass
            else:
                raise ValueError('joint type ' + j.tag)
            self.joints.append(J)
        self.joint_by_child = {J.child: J for J in self.joints}

    # ------------------------------------------------------------------ FK
    def joint_xform(self, J: Joint, q: dict):
        """X_parentbody→childbody = X_p · X_joint(q) · X_c^{-1}; returns (R, t)."""
        R = np.eye(3)
        t = np.zeros(3)
        for ax in J.axes:
            val = ax.func(q.get(ax.coord, 0.0)) if ax.coord else ax.func(0.0)
            if ax.kind == 'rot':
                R = R @ axis_angle(ax.axis, val)
            else:
                t = t + val * ax.axis / np.linalg.norm(ax.axis)
        Rj, tj = R, t
        # X_p * X_joint * X_c^-1
        R_pc = J.R_p @ Rj @ J.R_c.T
        t_pc = J.t_p + J.R_p @ (tj - Rj @ J.R_c.T @ J.t_c)
        return R_pc, t_pc

    def fk(self, q: dict):
        """모든 바디의 (R, t) in ground."""
        out = {'ground': (np.eye(3), np.zeros(3))}
        pending = list(self.joints)
        while pending:
            rest = []
            for J in pending:
                if J.parent in out:
                    Rp, tp = out[J.parent]
                    R, t = self.joint_xform(J, q)
                    out[J.child] = (Rp @ R, tp + Rp @ t)
                else:
                    rest.append(J)
            if len(rest) == len(pending):
                raise RuntimeError('unresolved ' + str([J.name for J in rest]))
            pending = rest
        return out

    def point_world(self, fk, body, p):
        R, t = fk[body]
        return t + R @ np.asarray(p)


def read_mot(path):
    rows = [l.split() for l in open(path) if l.strip()]
    i = [k for k, r in enumerate(rows) if r[0] == 'time'][0]
    hdr = rows[i]
    data = np.array([[float(x) for x in r] for r in rows[i + 1:]])
    return hdr, data


def mot_to_qdicts(hdr, data):
    return [dict(zip(hdr, row)) for row in data]
