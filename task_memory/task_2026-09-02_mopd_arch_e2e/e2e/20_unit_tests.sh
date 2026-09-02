#!/usr/bin/env bash
# Open-MOPD E2E — Step 20: run the Open-MOPD test suite.
#
# These tests ARE the executable specification of the MT-OPD semantics. They are
# pure-torch/numpy and need no GPU, no ray cluster, and no network — but they do
# import verl, which imports ray at verl/protocol.py:29, so the package must be
# installed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

banner "20 — Open-MOPD unit tests (executable spec)"

# Open-MOPD-specific tests, in dependency order (kernels -> reward -> wiring -> routing)
MOPD_TESTS=(
    tests/test_mt_opd_domain_metrics.py
    tests/test_mt_opd_gradient_share.py
    tests/test_mt_opd_m2_m3.py
    tests/test_delta_opd_reward.py
    tests/test_exopd_reward.py
    tests/test_exopd_trainer_wiring.py
    tests/trainer/ppo/test_mt_opd_batch_routing.py
    tests/test_domain_weighted_sampler.py
    tests/test_mix_rl_dispatch.py
    tests/test_hybrid_grpo_dense_topk.py
    tests/test_dp_actor_sp_topk_alignment.py
    tests/test_aime_average_metrics.py
    tests/trainer/ppo/test_kl_loss_coef_controller.py
)

cd "$REPO/training/verl"
LOG="$LOGS/20_unit_tests.log"
set +e
"$PY" -m pytest -v --no-header -p no:cacheprovider "${MOPD_TESTS[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[20] pytest exit=$rc  log=$LOG"

banner "20b — repo-level tests (experiments + evals)"
cd "$REPO"
LOG2="$LOGS/20b_repo_tests.log"
set +e
"$PY" -m pytest -v --no-header -p no:cacheprovider \
    experiments/tests \
    evals/verifier/tests \
    evals/rollout_engine/tests 2>&1 | tee "$LOG2"
rc2=${PIPESTATUS[0]}
set -e
echo "[20b] pytest exit=$rc2  log=$LOG2"

exit $(( rc != 0 || rc2 != 0 ))
