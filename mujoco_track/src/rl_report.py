# -*- coding: utf-8 -*-
"""RL 학습 곡선 그림 + 최종 정책 평가(하네스 yz / 없음). 사용: python3 -I rl_report.py <rl_dir> <outdir>"""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f_ in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f_:
        font_manager.fontManager.addfont(f_)
plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
CR, BL, MU = '#9B1B30', '#2F5D8F', '#6E6870'
rl_dir, outdir = sys.argv[1:3]
hist = json.load(open(f'{rl_dir}/eval_hist.json'))
base = json.load(open(f'{rl_dir}/baseline.json'))
steps = [h['steps'] / 1e6 for h in hist]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
ax = axes[0]
ax.plot(steps, [h['time_s'] for h in hist], color=CR, marker='o', ms=3, label='잔차 정책(결정적)')
ax.axhline(base['time_s'], color='k', ls='--', lw=1, label=f"명목 제어기 {base['time_s']:.2f} s")
ax.axhline(3 * 1.052, color=MU, ls=':', lw=1, label='에피소드 상한 3주기 (3.16 s)')
ax.set_title('넘어지기 전 생존 시간 (s) vs 학습 스텝 (M)', fontsize=10); ax.set_xlabel('PPO 스텝 (M)'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax = axes[1]
ax.plot(steps, [h['rmse_deg'] for h in hist], color=CR, marker='o', ms=3)
ax.axhline(base['rmse_deg'], color='k', ls='--', lw=1, label=f"명목 {base['rmse_deg']:.1f}°")
ax.set_title('관절 추종 RMSE (°)', fontsize=10); ax.set_xlabel('PPO 스텝 (M)'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax = axes[2]
ax.plot(steps, [h['harness_BW_rms'] for h in hist], color=CR, marker='o', ms=3, label='하네스 rms (BW)')
ax.plot(steps, [h['vx_mean'] for h in hist], color=BL, marker='s', ms=3, label='전진 속도 (m/s)')
ax.axhline(1.265, color=BL, ls=':', lw=1)
ax.set_title('하네스 지지력 · 전진 속도', fontsize=10); ax.set_xlabel('PPO 스텝 (M)'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle('잔차 RL(PPO) 학습 곡선 — 명목 제어기 위 케이블 활성 잔차 16 + 관절 목표 잔차 8, 하네스 = 높이·좌우·자세', fontsize=11)
fig.tight_layout(); fig.savefig(f'{outdir}/fig_rl_curve.png', dpi=150)
print('saved', len(hist), 'evals; last', hist[-1])
