# -*- coding: utf-8 -*-
"""PPO 잔차 정책 학습 (SB3). 사용: python3 -I train_ppo.py <outdir> <total_steps> [harness=yz|none]"""
import sys, os, json, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np
import torch
torch.set_num_threads(1)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from rl_env import ResidualWalkEnv, evaluate

outdir = sys.argv[1]
total = int(sys.argv[2])
harness = sys.argv[3] if len(sys.argv) > 3 else 'yz'
harness_arg = harness if harness != 'none' else False
os.makedirs(outdir, exist_ok=True)


def make(seed):
    def _f():
        return ResidualWalkEnv(harness=harness_arg, max_cycles=3.0, seed=seed)
    return _f


class EvalCB(BaseCallback):
    def __init__(self, every=50_000):
        super().__init__(); self.every = every; self.last = 0; self.hist = []

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            pol = lambda o: self.model.predict(o, deterministic=True)[0]
            res, _ = evaluate(pol, harness=harness_arg, cycles=3.0)
            res['steps'] = int(self.num_timesteps); res['wall'] = time.time() - T0
            self.hist.append(res)
            json.dump(self.hist, open(f'{outdir}/eval_hist.json', 'w'), indent=1)
            self.model.save(f'{outdir}/ppo_latest')
            print(f"[eval] steps {res['steps']} time {res['time_s']:.2f}s fell {res['fell']} rmse {res['rmse_deg']:.1f} harness {res['harness_BW_rms']:.2f} vx {res['vx_mean']:.2f} dist {res['dist_m']:.2f} wall {res['wall']/60:.1f} min", flush=True)
        return True


if __name__ == '__main__':
    T0 = time.time()
    venv = VecMonitor(SubprocVecEnv([make(i) for i in range(2)]))
    model = PPO('MlpPolicy', venv, n_steps=1024, batch_size=256, n_epochs=8, learning_rate=3e-4, gamma=0.98, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0), verbose=0, seed=0, device='cpu')
    base, _ = evaluate(None, harness=harness_arg, cycles=3.0)
    print('nominal baseline:', base, flush=True)
    json.dump(base, open(f'{outdir}/baseline.json', 'w'), indent=1)
    model.learn(total_timesteps=total, callback=EvalCB())
    model.save(f'{outdir}/ppo_final')
    pol = lambda o: model.predict(o, deterministic=True)[0]
    res, log = evaluate(pol, harness=harness_arg, cycles=3.0)
    print('final:', res, flush=True)
    json.dump(res, open(f'{outdir}/final.json', 'w'), indent=1)
    np.savez(f'{outdir}/final_log.npz', **log)
