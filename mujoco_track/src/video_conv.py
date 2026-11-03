# -*- coding: utf-8 -*-
"""영상 ③ — MuJoCo 변환 모델 재생: subject3 걷기·스쿼트 IK, 케이블 16개, 무릎 종속축 등식.
오른쪽: 케이블 길이 MuJoCo(실선) vs numpy 독립 FK(점) · 무릎 종속축 스플라인(OpenSim) vs 4차 다항식(MuJoCo 등식).
사용: xvfb-run -a -s "-screen 0 1280x720x24" env MUJOCO_GL=glfw python3 -I video_conv.py <outdir>
"""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from common import Leg16, load_trial, lowpass, COORDS
from vid_common import MjView, FrameWriter, fig_frame, setup_fonts, CABLE_COLORS, CR, BL, MU, INK
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

outdir = sys.argv[1]
R = DATA
OSIM, PJ, DATA = f'{R}/data/subject3/subject3_scaled.osim', f'{R}/report/newmus_paths_k8.json', f'{R}/data/subject3'
XML = model('leg16.xml')
setup_fonts()
L = Leg16(XML, OSIM, PJ)
M, P, meta = L.M, L.P, L.meta
view = MjView(XML, w=900, h=800, tendon_width=0.0085)
val = json.load(open(result('validate_conv.json')))['summary']


def np_tendon_lengths(q):
    fk = M.fk(q)
    out = []
    for mu in P['muscles']:
        for s in ('r', 'l'):
            pts = []
            for bname, pt in zip(mu['chain'], mu['points']):
                bn = bname if bname == 'pelvis' else bname[:-2] + '_' + s
                pt = np.array(pt, float)
                if s == 'l':
                    pt = pt * np.array([1, 1, -1])
                pts.append(M.point_world(fk, bn, pt))
            out.append(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))
    return np.array(out)


def segment(trial, t0, t1, speed, title):
    hi, di, hg, dg = load_trial(DATA, trial)
    fs = 1.0 / np.mean(np.diff(di[:, 0]))
    dif = di.copy(); dif[:, 1:] = lowpass(di[:, 1:], fs, 6.0)
    sel = (dif[:, 0] >= t0) & (dif[:, 0] <= t1)
    rows = dif[sel]
    T = rows[:, 0] - rows[0, 0]
    Q = []; Lmj = []; Lnp = []; knee = []
    for row in rows:
        q = dict(zip(hi, row))
        qp = L.qpos_from_coords(q); Q.append(qp)
        L.d.qpos[:] = qp; mujoco.mj_forward(L.m, L.d)
        Lmj.append(L.d.ten_length.copy()); Lnp.append(np_tendon_lengths(q)); knee.append(q['knee_angle_r'])
    return {'hi': hi, 'rows': rows, 'T': T, 'Q': np.array(Q), 'Lmj': np.array(Lmj), 'Lnp': np.array(Lnp), 'knee': np.array(knee), 'speed': speed, 'title': title, 'trial': trial}


segs = [segment('walking1', 0.0, 1.30, 1 / 3, '걷기 walking1 (IK 6 Hz, 1.3 s)'), segment('squats1', 2.1, 4.6, 1 / 2, '스쿼트 squats1 (2.5 s)')]
print('segments ready', [len(s['T']) for s in segs])

# ---------------- 무릎 종속축: OpenSim 스플라인 vs MuJoCo 다항식 등식
ka = np.linspace(0.0, 2.2, 120)
kp = meta['knee_poly']['walker_knee_r']
dep = {}
for lab, nm, scale in [('rotation3', '정강이 내회전 (°)', 180 / math.pi), ('rotation2', '정강이 내외반 (°)', 180 / math.pi), ('translation1', '병진 1 (mm)', 1000), ('translation2', '병진 2 (mm)', 1000)]:
    f = L.knee_axes['r'][lab].func
    spl = np.array([f(a) for a in ka]) * scale
    poly = np.polyval(kp[lab]['coef'][::-1], ka) * scale
    dep[lab] = (nm, spl, poly)

# ---------------- 그림 레이아웃
W, H, FPS = 1540, 924, 30
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
gs = GridSpec(2, 2, width_ratios=[1.12, 1], height_ratios=[1, 1], left=0.015, right=0.985, top=0.9, bottom=0.07, wspace=0.14, hspace=0.42)
ax_img = fig.add_subplot(gs[:, 0]); ax_img.axis('off')
ax_len = fig.add_subplot(gs[0, 1]); ax_knee = fig.add_subplot(gs[1, 1])
sup = fig.suptitle('', fontsize=13, color=INK, y=0.965)
im = ax_img.imshow(np.zeros((view.h, view.w, 3), np.uint8))
txt = ax_img.text(0.02, 0.975, '', transform=ax_img.transAxes, fontsize=11, va='top', color=INK)
foot = ax_img.text(0.02, 0.02, f"검증 211 자세 (걷기 132 · 스쿼트 63 · 가동범위 꼭짓점 16): 사이트 위치 오차 {val['site_pos_max_err_m']:.1e} m · 텐던 길이 {val['tendon_len_max_err_m']:.1e} m · 모멘트암 {val['moment_arm_max_err_m']:.1e} m",
                   transform=ax_img.transAxes, fontsize=9.5, color=MU)
names = [mu['name'] for mu in P['muscles']]
rgba_fixed = np.array([list(plt.matplotlib.colors.to_rgb(CABLE_COLORS[i // 2])) + [1.0] for i in range(16)])

# 무릎 패널 (고정)
lab_main = 'rotation3'
nm, spl, poly = dep[lab_main]
ax_knee.plot(np.degrees(ka), spl, color='k', lw=2.4, ls=':', label='OpenSim SimmSpline (원본)')
ax_knee.plot(np.degrees(ka), poly, color=CR, lw=1.6, label='MuJoCo 4차 다항식 등식 (변환)')
knee_pt, = ax_knee.plot([], [], 'o', color=CR, ms=9, zorder=5)
ax_knee.set_xlabel('무릎 굴곡각 (°)'); ax_knee.set_ylabel(nm)
ax_knee.set_title('무릎 종속축 4개 중 가장 큰 것 — 정강이 내회전(최대 15°): 스플라인 vs 다항식 (오차 ≤ 0.37 mrad)', fontsize=10.5)
ax_knee.grid(alpha=0.3); ax_knee.legend(fontsize=9, loc='upper left')
ax2 = ax_knee.inset_axes([0.62, 0.12, 0.35, 0.42])
nm2, spl2, poly2 = dep['translation2']
ax2.plot(np.degrees(ka), spl2, color='k', lw=1.8, ls=':'); ax2.plot(np.degrees(ka), poly2, color=CR, lw=1.2)
ax2.set_title('병진 2 (mm, 최대 6.4)', fontsize=8.5); ax2.tick_params(labelsize=7.5); ax2.grid(alpha=0.3)
knee_pt2, = ax2.plot([], [], 'o', color=CR, ms=6)

wr = FrameWriter(f'{outdir}/video_mj_conv.mp4', W, H, FPS)
poster_saved = False
n_total = 0
for si, S in enumerate(segs):
    T, Q, Lmj, Lnp = S['T'], S['Q'], S['Lmj'], S['Lnp']
    # 길이 패널 다시 그림 (오른다리 8개)
    ax_len.clear()
    for i in range(8):
        ax_len.plot(T, Lmj[:, 2 * i] * 100, color=CABLE_COLORS[i], lw=1.6, label=names[i])
        ax_len.plot(T[::4], Lnp[::4, 2 * i] * 100, 'o', color=CABLE_COLORS[i], ms=3.2, mfc='white', mew=1.0)
    cur = ax_len.axvline(0, color='k', lw=1)
    ax_len.set_xlim(T[0], T[-1]); ax_len.set_xlabel('time (s)'); ax_len.set_ylabel('케이블 길이 (cm)')
    ax_len.set_title(f'오른다리 케이블 8개 길이 — MuJoCo 엔진(실선) vs numpy 독립 순기구학(○) · 최대 차이 {val["tendon_len_max_err_m"]:.0e} m', fontsize=10.5)
    lo, hi_ = Lmj[:, ::2].min() * 100, Lmj[:, ::2].max() * 100
    ax_len.set_ylim(lo - 4, hi_ + 11)
    ax_len.grid(alpha=0.3); ax_len.legend(fontsize=8, ncol=8, loc='upper center', framealpha=0.9, handlelength=1.4, columnspacing=0.9)
    sup.set_text(f"MuJoCo 변환 모델(leg16.xml) — subject3 {S['title']} 재생 · 케이블 16개 = 스패셜 텐던, 무릎 종속축 = 등식 구속   {'1/3' if S['speed'] < 0.4 else '1/2'} 배속")
    n_frames = int(round((T[-1] / S['speed']) * FPS))
    reps = 2 if si == 0 else 1
    for rep in range(reps):
        for k in range(n_frames):
            t = k / FPS * S['speed']
            j = int(np.argmin(np.abs(T - t)))
            q = dict(zip(S['hi'], S['rows'][j]))
            view.set_camera([q['pelvis_tx'] + 0.05, 0.82, 0.0], 25, 9, 2.25)
            img = view.render(Q[j], tendon_rgba=rgba_fixed)
            im.set_data(img)
            txt.set_text(f"t = {t:.2f} s" + (f"   (반복 {rep + 1}/{reps})" if reps > 1 else ''))
            cur.set_xdata([t, t])
            kd = math.degrees(S['knee'][j])
            knee_pt.set_data([kd], [np.interp(S['knee'][j], ka, spl)]); knee_pt2.set_data([kd], [np.interp(S['knee'][j], ka, spl2)])
            fr = fig_frame(fig)
            wr.write(fr); n_total += 1
            if not poster_saved and si == 0 and t >= 0.55:
                plt.imsave(f'{outdir}/video_mj_conv_poster.png', fr); poster_saved = True
    print('segment', si, 'frames', n_frames * reps)
wr.close()
print('saved', n_total, 'frames')
