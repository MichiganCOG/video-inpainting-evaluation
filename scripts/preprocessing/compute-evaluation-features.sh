#!/bin/bash

### Kill pre-processing scripts if Ctrl-C is pressed ###

function kill_subprocesses {
    kill "$compute_flow_occlusion_pid"
    kill "$compute_fid_features_pid"
    kill "$compute_vfid_features_pid"
    kill "$compute_vfid_clips_features_pid"
}
trap kill_subprocesses INT

### Collect arguments ###

if [ "$#" != 2 ]; then
    echo "Usage: compute-evaluation-features.sh GT_FRAMES_ROOT EVAL_FEATURES_DEST_ROOT"
    exit 1
fi
GT_FRAMES_ROOT="$1"
EVAL_FEATURES_DEST_ROOT="$2"

### Run scripts ###

python -m src.main.compute_flow_occlusion \
    --gt_root "$GT_FRAMES_ROOT" \
    --output_root "$EVAL_FEATURES_DEST_ROOT" \
    &
compute_flow_occlusion_pid="$!"

python -m src.main.compute_fid_features \
    --gt_root "$GT_FRAMES_ROOT" \
    --output_path "$EVAL_FEATURES_DEST_ROOT/fid.npy" \
    &
compute_fid_features_pid="$!"

python -m src.main.compute_vfid_features \
    --gt_root "$GT_FRAMES_ROOT" \
    --output_path "$EVAL_FEATURES_DEST_ROOT/vfid.npy" \
    &
compute_vfid_features_pid="$!"

python -m src.main.compute_vfid_clips_features \
    --gt_root "$GT_FRAMES_ROOT" \
    --output_path "$EVAL_FEATURES_DEST_ROOT/vfid_clips.npy" \
    &
compute_vfid_clips_features_pid="$!"

### Wait for scripts to finish ###

wait "$compute_flow_occlusion_pid"
wait "$compute_fid_features_pid"
wait "$compute_vfid_features_pid"
wait "$compute_vfid_clips_features_pid"
