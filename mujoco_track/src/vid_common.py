# -*- coding: utf-8 -*-
"""영상 공통 — MuJoCo 오프스크린 렌더(고정 카메라, Y-up 월드), 추가 지오메트리(캡슐·화살표), ffmpeg 파이프 기록, 글꼴.
렌더는 Xvfb + GLFW(Mesa llvmpipe)로 돈다:  xvfb-run -a -s "-screen 0 1280x720x24" env MUJOCO_GL=glfw python3 -I <script>
"""
import re, math, subprocess
import numpy as np, mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

CR, CRD, BL, GR, MU, GOLD, INK, LN = '#9B1B30', '#731122', '#2F5D8F', '#2E7D4F', '#6E6870', '#B8860B', '#26262B', '#E3DADC'
CABLE_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']   # M0~M7 (학생 영상과 같은 tab10)
ACT_CMAP = LinearSegmentedColormap.from_list('act', ['#BFD7EA', '#E8A0A8', '#9B1B30', '#5A0D1B'])


def setup_fonts():
    for f_ in font_manager.findSystemFonts():
        if 'NotoSansCJK' in f_:
            font_manager.fontManager.addfont(f_)
    plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def quat_from_mat(R):
    """회전행렬 → 쿼터니언 (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


class MjView:
    """시뮬 모델과 같은 XML에 고정 카메라만 더한 '보기용' 모델. qpos 를 복사해 렌더한다 (Y-up 월드)."""

    def __init__(self, xml_path=None, xml_string=None, w=900, h=760, tendon_width=0.007, shadow=True):
        txt = open(xml_path).read() if xml_string is None else xml_string
        cam = '<camera name="vcam" pos="0 1 3" xyaxes="1 0 0 0 1 0"/>'
        txt = txt.replace('<worldbody>', '<worldbody>\n    ' + cam, 1)
        if '<global ' in txt:
            txt = re.sub(r'<global offwidth="\d+" offheight="\d+"/>', f'<global offwidth="{max(w, 1280)}" offheight="{max(h, 720)}"/>', txt)
        else:
            txt = txt.replace('<worldbody>', f'<visual><global offwidth="{max(w, 1280)}" offheight="{max(h, 720)}"/></visual>\n  <worldbody>', 1)
        # 흰 배경(skybox flat) + 바닥 체커 재질
        asset = ('<asset><texture name="vsky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="32" height="32"/>'
                 '<texture name="vgrid" type="2d" builtin="checker" rgb1="0.94 0.94 0.95" rgb2="0.87 0.87 0.89" width="512" height="512"/>'
                 '<material name="vgrid" texture="vgrid" texrepeat="2 2" texuniform="true" reflectance="0"/></asset>\n  ')
        txt = txt.replace('<worldbody>', asset + '<worldbody>', 1)
        txt = re.sub(r'<geom name="floor" ', '<geom name="floor" material="vgrid" ', txt)
        # 몸통 척추·머리 (보기용)
        txt = re.sub(r'(<body name="torso"[^>]*>)', r'\1\n<geom type="capsule" fromto="0 0 0 0 0.44 0" size="0.028"/><geom type="sphere" pos="0.01 0.56 0" size="0.085"/>', txt, count=1)
        self.m = mujoco.MjModel.from_xml_string(txt)
        self.d = mujoco.MjData(self.m)
        self.cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, 'vcam')
        self.w, self.h = w, h
        self.renderer = mujoco.Renderer(self.m, h, w)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadow else 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        if self.m.ntendon:
            self.m.tendon_width[:] = tendon_width
        self.bid = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(self.m.nbody)}

    def set_camera(self, lookat, az_deg=35.0, el_deg=12.0, dist=3.0, up=(0, 1, 0)):
        az, el = math.radians(az_deg), math.radians(el_deg)
        dvec = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        pos = np.asarray(lookat, float) + dist * dvec
        z = pos - np.asarray(lookat, float); z /= np.linalg.norm(z)
        x = np.cross(np.asarray(up, float), z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        self.m.cam_pos[self.cid] = pos
        self.m.cam_quat[self.cid] = quat_from_mat(R)

    def render(self, qpos, extra=(), tendon_rgba=None):
        m, d = self.m, self.d
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        if tendon_rgba is not None:
            m.tendon_rgba[:] = tendon_rgba
        self.renderer.update_scene(d, camera='vcam')
        scn = self.renderer.scene
        for g in extra:
            if scn.ngeom >= scn.maxgeom:
                break
            geom = scn.geoms[scn.ngeom]
            rgba = np.array(g.get('rgba', (0.5, 0.5, 0.5, 1)), np.float32)
            kind = g['type']
            if kind == 'sphere':
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([g['size'], 0, 0], float), np.asarray(g['pos'], float), np.eye(3).flatten(), rgba)
            else:
                gt = {'capsule': mujoco.mjtGeom.mjGEOM_CAPSULE, 'arrow': mujoco.mjtGeom.mjGEOM_ARROW, 'line': mujoco.mjtGeom.mjGEOM_LINE, 'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER}[kind]
                mujoco.mjv_initGeom(geom, gt, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
                mujoco.mjv_connector(geom, gt, float(g['width']), np.asarray(g['from'], float), np.asarray(g['to'], float))
            scn.ngeom += 1
        return self.renderer.render()


def act_rgba(a):
    """활성 0~1 → 텐던 색 (연파랑 → 진홍)."""
    c = ACT_CMAP(np.clip(a, 0, 1))
    return np.array(c, float)


class FrameWriter:
    def __init__(self, path, w, h, fps=30, crf=20):
        self.w, self.h = w, h
        self.p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                                   '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', 'medium', '-movflags', '+faststart', path], stdin=subprocess.PIPE)
        self.n = 0

    def write(self, frame):
        assert frame.shape[0] == self.h and frame.shape[1] == self.w, (frame.shape, self.h, self.w)
        self.p.stdin.write(np.ascontiguousarray(frame[..., :3]).tobytes()); self.n += 1

    def close(self):
        self.p.stdin.close(); self.p.wait()


def fig_frame(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def interp_rows(t, T, X):
    """시각 t 에서 행렬 X (len(T) × n) 선형보간."""
    return np.array([np.interp(t, T, X[:, j]) for j in range(X.shape[1])])


def ghost_capsules(view, qpos, rgba=(0.55, 0.55, 0.58, 0.45), width=0.022, chains=None):
    """기준 자세의 바디 원점을 잇는 반투명 캡슐 (고스트 뼈대). view.d 를 쓰므로 render 전에 호출해 extra 로 넘긴다."""
    m = view.m; d = mujoco.MjData(m)
    d.qpos[:] = qpos; mujoco.mj_kinematics(m, d)
    chains = chains or [['pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r'], ['pelvis', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l'], ['pelvis', 'torso']]
    out = []
    for ch in chains:
        pts = [d.xpos[view.bid[n]].copy() for n in ch]
        if ch[-1] == 'torso':
            Rt = d.xmat[view.bid['torso']].reshape(3, 3)
            pts.append(pts[-1] + Rt @ np.array([0.0, 0.42, 0.0]))
        for i in range(len(pts) - 1):
            if np.linalg.norm(pts[i + 1] - pts[i]) > 1e-4:
                out.append({'type': 'capsule', 'from': pts[i], 'to': pts[i + 1], 'width': width, 'rgba': rgba})
    return out


def foot_contact_arrows(m, d, bid_sets, scale=0.0004, rgba=(0.18, 0.55, 0.3, 1.0), width=0.012):
    """접촉력 합을 발별 화살표로 (접촉점 평균에서 힘 방향)."""
    out = []
    f6 = np.zeros(6)
    for bodies in bid_sets:
        F = np.zeros(3); P = np.zeros(3); n = 0
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.geom_bodyid[c.geom1]; b2 = m.geom_bodyid[c.geom2]
            if b1 in bodies or b2 in bodies:
                mujoco.mj_contactForce(m, d, i, f6)
                fr = c.frame.reshape(3, 3)
                fw = fr.T @ f6[:3]                 # 접촉 프레임 → 월드 (법선은 geom1 → geom2; 힘은 geom2 에 작용하는 방향)
                if b1 in bodies:
                    fw = -fw
                F += fw; P += c.pos; n += 1
        if n and np.linalg.norm(F) > 5:
            P /= n
            out.append({'type': 'arrow', 'from': P, 'to': P + scale * F, 'width': width, 'rgba': rgba})
    return out
