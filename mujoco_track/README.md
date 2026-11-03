# mujoco_track — OpenSim 설계안의 MuJoCo 변환 · 순동역학 보행 · 잔차 RL · 팔꿈치 sim-to-real 리허설

세종대 의료로봇연구실 · 미래판기술프로젝트(RS-2026-25541200) · 작성 2026-10-31 ~ 11-03

나유빈 학생 저장소([nauqin/Artificial-muscle-humanoid](https://github.com/nauqin/Artificial-muscle-humanoid))의 하지 설계안
(subject3 스케일 모델 + 케이블 16개 경로 json)을 **MuJoCo로 1:1 변환**하고, 그 위에서 역동역학 재생·순동역학 보행·잔차 RL·팔꿈치 리그
sim-to-real 리허설을 돌린 결과와 코드다. 11/2 전 기관 중간점검 화상회의 발표자료의 "10월 신규 ③~⑥"과 10장(상체 확장 입력) 영상이
여기서 나왔다. 학생 저장소의 9월 말 README 계획 중 "10월 하반 MuJoCo 전환 + RL 보행 검증", "11월 팔꿈치 sim-to-real" 항목에 해당한다.

> 한 줄 결론 — 변환은 수치 정밀도(1e-8 m)로 맞고 역동역학은 OpenSim과 1 % 이내. 순동역학 보행은 하네스 보조 아래 잔차 RL이 3주기를
> 유지하지만 **자립·자연스러운 보행은 아직 아니다**(원인: 접촉 비동기 위상·균형 제어 부재·케이블 포화). 팔꿈치 리허설은 README가 정한
> 1차 열지연 모델로는 폐루프 결론이 틀린다는 것(냉각 히스테리시스 필요)을 보였다.

## 폴더

```
mujoco_track/
├── README.md                 ← 이 문서
├── requirements.txt
├── run_all.sh                ← 전체 재현 순서 (AMH_DATA 필요)
├── src/                      ← 스크립트 (아래 표)
│   ├── paths.py              ← 경로 한 곳에서 관리 (AMH_DATA = 학생 저장소 루트)
│   └── diag/                 ← 질의응답 중 돌린 진단 스크립트
├── models/                   ← 변환된 MJCF: leg16.xml(검증용) · leg16_dyn.xml(동역학용) · moore9/moore3.xml(팔 포함, 케이블 없음) · *_meta.json
├── results/                  ← json/npz 결과, figures/ 그림, videos/ 발표 영상(mp4 + 포스터), rl_yz2/ PPO 정책
└── docs/                     ← 11/2 회의 최종 발표자료, 질의응답 정리, 11월 보행 개선 계획
```

| src/ 파일 | 하는 일 |
| --- | --- |
| `paths.py` | SRC/MODELS/RESULTS/DATA 경로. DATA는 `AMH_DATA` 환경변수 또는 `../Artificial-muscle-humanoid` |
| `osim_model.py` | .osim 파싱(바디·관절·오프셋 프레임·SpatialTransform 함수) + numpy 순기구학(검증 기준) |
| `build_mjcf.py` | .osim + 케이블 json + walking IK → MJCF. 무릎 종속축 4개 = 4차 다항식 등식, 발 접촉 구 5개, 케이블 = site/spatial tendon/general actuator. `--dyn` = 잠금 관절을 평균값으로 굳힌 동역학용(22 DOF). json에 `{"muscles": []}`를 주면 케이블 없는 모델(팔 포함 Moore2015용) |
| `validate_conv.py` | 211 자세(걷기 132 + 스쿼트 63 + ROM 꼭짓점 16)에서 사이트 위치·텐던 길이·모멘트암 MuJoCo vs numpy, 학생 json 교차, 그림 |
| `common.py` | `Leg16` 래퍼: 좌표→qpos, 유효 모멘트암(무릎 연쇄법칙), GRF→일반화힘(`mj_applyFT`), 필터, trial 로더 |
| `id_replay.py` | 역동역학 재생(`mj_inverse`) + 정적 케이블 검사(0≤f≤F_max, R f = 1.1τ) + 그림 |
| `walk_sim.py` | 순동역학: 기준 궤적(1주기 반복), SMA 1차 열지연, PD+ID 피드포워드 → 유계 최소자승 배분 → 열지연 보상, 골반 하네스(yz/full) |
| `rl_env.py` / `train_ppo.py` / `rl_eval.py` / `rl_report.py` | 잔차 RL 환경(관측 44 / 행동 24), PPO 학습, 8개 시작위상 평가 + 영상, 학습 곡선 |
| `walk_report.py` | 명목 제어기 3조건(이상 / SMA / SMA+SES) 그림 + 스틱피겨 영상 |
| `elbow_rig.py` | 팔꿈치 리그: 가상 실물(T3.1 3상태 SMA) vs README 1차 모델 vs 히스테리시스 보강안 — P4 식별, 개루프 검증, 폐루프 지표 |
| `mass_test.py` | 30 % 경량화 가설(질량·관성 λ배)을 명목 제어기·기존 정책으로 재평가 (조건 base/A/B/C/D) |
| `vid_common.py` | 영상 공통: MuJoCo 오프스크린 렌더(고정 카메라, Y-up), 고스트 뼈대·화살표, ffmpeg 파이프, 글꼴 |
| `video_upper.py` `video_conv.py` `video_id.py` `video_walk.py` `video_elbow.py` | 발표 영상 ①(상체 입력) ③(변환 재생) ④(역동역학) ⑤(보행 4조건 + 명목 vs 정책) ⑥(팔꿈치 리그 3개) |
| `diag/diag_ideal.py` | 이상 액추에이터가 왜 못 걷나: 3주기 시도 시 넘어지는 시각, 골반 높이, 포화 케이블, 기준 궤적 이음새 |
| `diag/elbow_why.py` | README 1차 모델이 폐루프에서 왜 진동하나: 냉각 시정수만 바꿔 비교 |

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # mujoco 3.x, gymnasium, stable-baselines3, torch, scipy, numpy, matplotlib, pillow
sudo apt install ffmpeg xvfb                # 영상 제작용 (GPU 없는 서버: Xvfb + Mesa 소프트웨어 렌더)
```

* 확인한 환경: Python 3.13, mujoco 3.17, stable-baselines3 2.x, CPU 2코어(학습 1.2 M 스텝 54분).
* 글꼴: 그림·영상의 한글은 Noto Sans CJK KR을 찾아 쓴다(없으면 □로 나옴 → `fonts-noto-cjk` 설치).

## 데이터 연결

스크립트는 학생 저장소의 파일을 직접 읽는다(복사하지 않음).

```bash
git clone https://github.com/nauqin/Artificial-muscle-humanoid ../Artificial-muscle-humanoid   # 또는 이미 있는 경로
export AMH_DATA=/path/to/Artificial-muscle-humanoid
python3 -I src/paths.py        # DATA 경로가 (exists)로 나오면 됨
```

읽는 파일: `data/subject3/subject3_scaled.osim`, `data/subject3/{walking1,squats1}_segment_0_{ik,grf}.mot`, `report/newmus_paths_k8.json`,
`data/moore2015/{subject3,subject9}/*_scaled.osim`, `data/moore2015/.../T0xx_segment_k_{ik,grf}.mot`, `out/moore2015/summary.csv`.

## 실행 순서

`run_all.sh`에 같은 순서가 들어 있다. 모든 스크립트는 `python3 -I`(사용자 site·현재 디렉터리 무시)로 돌려도 된다.

```bash
R=$AMH_DATA; S=src; M=models; O=results
# 1. 변환 (몇 초)
python3 -I $S/build_mjcf.py $R/data/subject3/subject3_scaled.osim $R/report/newmus_paths_k8.json $R/data/subject3/walking1_segment_0_ik.mot $M/leg16.xml
python3 -I $S/build_mjcf.py $R/data/subject3/subject3_scaled.osim $R/report/newmus_paths_k8.json $R/data/subject3/walking1_segment_0_ik.mot $M/leg16_dyn.xml --dyn
python3 -I $S/build_mjcf.py $R/data/moore2015/subject9/subject9_scaled.osim $M/nocables.json $R/data/moore2015/subject9/T026_segment_16_ik.mot $M/moore9.xml
python3 -I $S/build_mjcf.py $R/data/moore2015/subject3/subject3_scaled.osim $M/nocables.json $R/data/moore2015/subject3/T048_segment_3_ik.mot $M/moore3.xml
# 2. 검증 · 역동역학 (몇 십 초)
python3 -I $S/validate_conv.py $R/data/subject3/subject3_scaled.osim $R/report/newmus_paths_k8.json $M/leg16.xml $R/data/subject3/walking1_segment_0_ik.mot $R/data/subject3/squats1_segment_0_ik.mot $O
python3 -I $S/id_replay.py $M/leg16.xml $R/data/subject3/subject3_scaled.osim $R/report/newmus_paths_k8.json $R/data/subject3 walking1 0.248 1.121 $O        # 학생 ID 창
python3 -I $S/id_replay.py $M/leg16.xml $R/data/subject3/subject3_scaled.osim $R/report/newmus_paths_k8.json $R/data/subject3 walking1 0.0 1.30 $O/full    # 피드포워드용 전 구간
cp $O/full/id_walking1.npz $O/id_walking1_full.npz
# 3. 순동역학 3조건 (1~2분) · 잔차 RL (CPU 2코어 54분) · 평가
python3 -I $S/walk_report.py $O
python3 -I $S/train_ppo.py $O/rl_yz2 1200000 yz
python3 -I $S/rl_eval.py $O/rl_yz2/ppo_final.zip $O yz ; python3 -I $S/rl_report.py $O/rl_yz2 $O
# 4. 팔꿈치 리허설 (1~2분) · 경량화 가설 (1분)
python3 -I $S/elbow_rig.py $O
python3 -I $S/mass_test.py $O/rl_yz2/ppo_final.zip $O 0.7
# 5. 영상 (각 2~5분, 소프트웨어 렌더)
V='xvfb-run -a -s "-screen 0 1280x720x24" env MUJOCO_GL=glfw python3 -I'
eval $V $S/video_upper.py $O/videos; eval $V $S/video_conv.py $O/videos; eval $V $S/video_id.py $O/videos
eval $V $S/video_walk.py $O/videos $O/rl_yz2/ppo_final.zip; eval $V $S/video_elbow.py $O/videos
```

디스플레이가 있는 PC면 `xvfb-run` 없이 `MUJOCO_GL=glfw`, NVIDIA GPU 서버면 `MUJOCO_GL=egl`로 훨씬 빠르다.

## 결과 요약 (발표자료 10월 신규 ③~⑥ · 10장)

* **변환** — 211 자세에서 사이트 위치 7e-9 m · 텐던 길이 1e-8 m · 모멘트암 9e-9 m (MuJoCo vs numpy 독립 구현). 학생 OpenSim 값과 arm_min ≤ 1.5 mm. 무릎 종속축 다항식 오차 0.37 mrad / 0.004 mm.
* **역동역학** — 피크 Nm/kg 발목 1.401/1.301 · 무릎 0.407/0.475 · 고관절 0.957/0.854 (OpenSim 1.399/1.298 · 0.409/0.477 · 0.945/0.906). 골반 잔차 최대 4.8 %.
* **정적 케이블 검사** — 전 구간(0~1.3 s) 오른다리 최악 3.9 % · 왼다리 0.8 %(첫 검증); 학생 ID 창(0.25~1.12 s) 오른 4.5 % · 왼 0 %. 활성 상한 오른 M1·M2·M7 = 1.0, 왼 M1·M2 = 1.0.
* **순동역학 1주기(yz 하네스)** — 이상 액추에이터 RMSE 12.5° / SMA 지연 23.5°(1.0 s 넘어짐) / SMA+발목 강성 100 N·m/rad 16.5°. 단 "1주기 유지"는 넘어짐 판정(골반 < 0.6 m)이 안 뜬 것일 뿐 — 골반이 15~16 cm 가라앉고 계속 걸리면 이상 1.34 s · 이상+SES 1.32 s · SMA+SES 1.09 s에 넘어진다(`diag/diag_ideal.py`). 원인은 지지 다리 케이블(M0·M1·M2·M7) 포화 + 균형 제어 부재 + 시간 기반 위상.
* **잔차 RL(PPO 1.2 M 스텝)** — 시작 위상 0에서 3주기 유지(3.16 s), RMSE 13.2°, 하네스 0.19 BW (명목 1.09 s·16.8°·0.35 BW). 시작 위상 8개 평균 생존 0.99 → 2.53 s, 3/8 완주, 속도 0.90 m/s(안정을 위해 느려짐). 하네스 없이는 1.1 s 내 넘어짐. 절뚝이는 이유와 개선 순서는 `docs/plan_nov_walking.md`.
* **30 % 경량화 가설(`mass_test.py`)** — 질량·관성 0.7배를 공정하게(피드포워드·PD·강성·하네스도 0.7배 = 힘 1.43배 등가) 비교하면 포화 케이블 4 → 0개, 평균 활성 0.36 → 0.28이지만 완주율 불변(명목 0/8, 정책 3/8). 제어기를 그대로 두면 정책 6/8로 보이나 하네스가 상대적으로 43 % 세진 효과. 걷기의 시간 척도(√(L/g))는 질량과 무관.
* **팔꿈치 리허설** — README 1차 모델: 식별 잔차 1.4 %이지만 개루프 3.7~16.6°, 폐루프 한계 진동(정상 오차 45°). 보강안(1차 + 히스테리시스 맵) 0.35 % / 3.6~8.1° / 4.0°. −3 dB ≈ 2.7 Hz(±17°, 2 kg, 공랭, 열 상한), T_max 86~106 °C, 듀티 0.11~0.28. 냉각 시정수를 바꿔도 README 모델의 진동은 안 사라진다(`diag/elbow_why.py`) → 구조(히스테리시스) 문제.
* **상체 확장 입력(Moore2015)** — 학생 변환 결과(팔 포함 모델·IK·보정 GRF)를 MuJoCo로 옮겨 재생. MuJoCo 역동역학 잔차 subject9 1.2 m/s 최대 3.3 %(학생 2.0 %), subject3 1.6 m/s 7.0 %(5.6 %). 요추 신전 모멘트 피크 0.46~0.69 Nm/kg.

## 밟은 함정 (다시 밟지 않도록)

1. OpenSim 왼쪽 고관절 내전·회전은 축이 아니라 `LinearFunction(-1)`로 뒤집힌다 — 축에 슬로프를 곱해야 한다(안 하면 34° 거울상).
2. `mj_forward`가 `qacc`를 덮어쓴다 — 역동역학은 forward 뒤에 qacc를 넣고 `mj_inverse`.
3. `mj_inverse`의 `qfrc_inverse`는 외력을 빼지 않는다 — GRF 일반화힘(`mj_applyFT`)을 직접 빼야 관절 모멘트.
4. 무릎 모멘트암은 종속축(rotation2·3, translation1·2) 기여를 연쇄법칙으로 더해야 OpenSim과 같다(최대 19 mm 차이).
5. 가벼운 발(거골 0.09 kg)을 등식 구속으로 잠그면 접촉 충격에 발산 — 동역학용은 잠금 관절을 바디에 굳힌다(`--dyn`).
6. 발 관성 0.015 kg·m²에서 200 Hz 영차 홀드 PD는 Kd ≤ 2 (Kd 6이면 수치 발산). 실물도 모듈 로컬 1 kHz 루프가 필요하다.
7. 3상태 SMA 모델의 가열/냉각 판정은 반전점 온도가 아니라 직전 온도와 비교해야 한다. 냉각 ξ 식은 ½[1 − cos].
8. 짧은(1 s) 조각에 필터를 걸고 역동역학을 하면 가장자리 0.1 s가 잔차를 지배한다 — 통계는 가장자리를 빼고 낸다(`video_upper.py`).
9. MuJoCo 오프스크린 렌더러(GLFW)는 프로세스당 하나만 — 두 번째 모델을 그리려면 먼저 `renderer.close()`.
10. "완주"라는 말을 쓰지 말 것 — 넘어짐 플래그가 안 떴다는 것과 걷기 성공은 다르다. 골반 높이·하네스 힘·계속 걸렸을 때 넘어지는 시각을 같이 적는다.

## 다음

* 보행: `docs/plan_nov_walking.md` — 접촉 동기 위상 → 균형 피드백 → RL 환경 재설계·재학습(연구실 GPU) → 여유율·SES. 경로 3차(H1) 결과가 나오면 `build_mjcf.py` 재실행 → 같은 검증 자동.
* 팔꿈치: 실물 식별 항목에 냉각 문턱(Ms/Mf) 추가, PRBS 개루프 검증을 합격 기준에, 힘/저항 피드백. 리그가 오면 `elbow_rig.py`의 `SMA3`를 측정 데이터로 교체.
* 상체: Moore2015 모델은 운동학·역동역학 입력까지. 팔 케이블은 U5~U6 뒤에.

## 함께 보기

* `docs/QA_notes.md` — 발표 준비 중 나온 질문과 답(ID 토크, MJCF, 하네스, RMSE, 경량화, 절뚝임, 팔꿈치 진동 원인 등).
* `docs/미래판기술_중간점검화상회의_20261102_세종대_발표자료_최종.pptx` — 11/2 회의에서 실제 발표한 최종본(19장, 영상 7편 포함). 본문만 남긴 것이라 부록 A~L(검산 기준·데이터 출처·EMG·경로 상세·Moore2015 상세·코드 지도·MJCF 구조·순동역학 표·팔꿈치 표·RL 평가)은 들어 있지 않다 — 그 내용은 이 README와 `docs/QA_notes.md`에 있다. 14장 표의 "완주"는 `docs/QA_notes.md` 7절의 정정(유지* = 넘어짐 판정이 안 뜬 것일 뿐)을 적용해 읽을 것..
