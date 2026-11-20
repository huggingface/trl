#!/bin/bash
# Inside the miles v0.2.0 container: the example's launcher, unchanged, `prepare` then `train`.
set -uo pipefail
cd /root/miles
git -C /root/miles log -1 --format='miles %h %cd' 2>/dev/null | tee -a $RUN/times.txt
ls examples/text_to_sql/ | tee -a $RUN/times.txt
L=examples/text_to_sql/run_qwen3_6_35b_a3b_text_to_sql.py
COMMON="--model-name Qwen3.5-35B-A3B --megatron-model-type qwen3.5-35B-A3B --model-dir $WORK/models --data-dir $WORK/datasets --output-dir $WORK/out"
echo "$(date -u) prepare" | tee -a $RUN/times.txt
python $L prepare $COMMON > $RUN/prepare.log 2>&1 || { echo "$(date -u) prepare failed" | tee -a $RUN/times.txt; tail -n 40 $RUN/prepare.log; exit 1; }
echo "$(date -u) train" | tee -a $RUN/times.txt
python $L train $COMMON --extra-args "--attention-backend unfused" > $RUN/train.log 2>&1
rc=$?
echo "$(date -u) train exit $rc" | tee -a $RUN/times.txt
tail -n 60 $RUN/train.log
exit $rc
