#!/usr/bin/env bash
# 전체 재현 순서. 사용: AMH_DATA=/path/to/Artificial-muscle-humanoid ./run_all.sh [--no-rl] [--no-video]
set -euo pipefail
cd "$(dirname "$0")"
: "${AMH_DATA:?AMH_DATA (학생 저장소 루트)를 지정하세요}"
R=$AMH_DATA; S=src; M=models; O=results
NO_RL=0; NO_VIDEO=0
for a in "$@"; do [ "$a" = "--no-rl" ] && NO_RL=1; [ "$a" = "--no-video" ] && NO_VIDEO=1; done
mkdir -p $O/full $O/videos $O/figures
python3 -I $S/paths.py

echo "== 1. 변환"
python3 -I $S/build_mjcf.py $R/data/subject3/subject3_scaled.osim $R/report/newmus_paths_k8.json $R/data/subject3/walking1_segment_0_ik.mot $M/leg16.xml
python3 -I $S/build_mjcf.py $R/data/subject3/subject3_scaled.osim $R/report/newmus_paths_k8.json $R/data/subject3/walking1_segment_0_ik.mot $M/leg16_dyn.xml --dyn
python3 -I $S/build_mjcf.py $R/data/moore2015/subject9/subject9_scaled.osim $M/nocables.json $R/data/moore2015/subject9/T026_segment_16_ik.mot $M/moore9.xml
python3 -I $S/build_mjcf.py $R/data/moore2015/subject3/subject3_scaled.osim $M/nocables.json $R/data/moore2015/subject3/T048_segment_3_ik.mot $M/moore3.xml

echo "== 2. 검증 · 역동역학"
python3 -I $S/validate_conv.py $R/data/subject3/subject3_scaled.osim $R/report/newmus_paths_k8.json $M/leg16.xml $R/data/subject3/walking1_segment_0_ik.mot $R/data/subject3/squats1_segment_0_ik.mot $O
python3 -I $S/id_replay.py $M/leg16.xml $R/data/subject3/subject3_scaled.osim $R/report/newmus_paths_k8.json $R/data/subject3 walking1 0.248 1.121 $O
python3 -I $S/id_replay.py $M/leg16.xml $R/data/subject3/subject3_scaled.osim $R/report/newmus_paths_k8.json $R/data/subject3 walking1 0.0 1.30 $O/full
cp $O/full/id_walking1.npz $O/id_walking1_full.npz

echo "== 3. 순동역학 3조건"
python3 -I $S/walk_report.py $O
if [ $NO_RL -eq 0 ]; then
  echo "== 3b. 잔차 RL (CPU 2코어 기준 약 1시간)"
  python3 -I $S/train_ppo.py $O/rl_yz2 1200000 yz
  python3 -I $S/rl_eval.py $O/rl_yz2/ppo_final.zip $O yz
  python3 -I $S/rl_report.py $O/rl_yz2 $O
fi

echo "== 4. 팔꿈치 리허설 · 경량화 가설"
python3 -I $S/elbow_rig.py $O
python3 -I $S/mass_test.py $O/rl_yz2/ppo_final.zip $O 0.7

if [ $NO_VIDEO -eq 0 ]; then
  echo "== 5. 영상 (소프트웨어 렌더, 각 2~5분)"
  V="xvfb-run -a -s '-screen 0 1280x720x24' env MUJOCO_GL=glfw python3 -I"
  eval $V $S/video_upper.py $O/videos
  eval $V $S/video_conv.py $O/videos
  eval $V $S/video_id.py $O/videos
  eval $V $S/video_walk.py $O/videos $O/rl_yz2/ppo_final.zip
  eval $V $S/video_elbow.py $O/videos
fi
mv -f $O/fig_*.png $O/figures/ 2>/dev/null || true
echo "done"
