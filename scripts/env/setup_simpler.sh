#!/usr/bin/env bash
# SIMPLER (SimplerEnv + ManiSkill2_real2sim; sapien 2.2 + Vulkan) in its own environment.
#   bash scripts/env/setup_simpler.sh [env_name]
# X-VLA can run SIMPLER in its own environment (setup_xvla.sh); this environment is for serving the simulator
# separately with `qbdvla run --model remote`.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-simpler}"

conda_env "$ENV" 3.10
setup_simpler "$ENV"
install_qbdvla "$ENV"
echo "done: qbdvla run --model remote --model-kwargs url=http://127.0.0.1:8010 --benchmark simpler --suite google_robot_va"
