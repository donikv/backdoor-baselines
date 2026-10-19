#!/bin/bash
# Third-party code used by bdbaselines, fetched at the pinned commits (not distributed here).
#   bash scripts/fetch_third_party.sh               FreeEagle (MIT), needed for FreeEagle scores
#   bash scripts/fetch_third_party.sh --validation  also DFBScanner, MM-BD and BackdoorBench's clp.py, for bdbaselines.validate
set -euo pipefail
D=third_party
mkdir -p "$D"
clone() {
    [ -d "$D/$1/.git" ] || git clone -q "$2" "$D/$1"
    git -C "$D/$1" checkout -q "$3"
    echo "$1 at $3"
}
clone FreeEagle https://github.com/FuChong-cyber/Data-Free-Neural-Backdoor-Detector-FreeEagle d04582206d4cd32c0f9118cd49de7e7fbad93f54
if [ "${1:-}" = "--validation" ]; then
    clone DFBScanner https://github.com/EboYu/DFBScanner cc8fdd8c8b03cbb25029c017f07261b4f25701a3
    clone MM-BD https://github.com/wanghangpsu/MM-BD 903338d69162504767d947d5b923cbf36d497765
    mkdir -p "$D/backdoorbench"
    curl -sSfL https://raw.githubusercontent.com/SCLBD/BackdoorBench/3f4b1a92ed6c13e0cd21534df56155cf3ef71f87/defense/clp.py \
        -o "$D/backdoorbench/clp.py"
    echo "backdoorbench/clp.py at 3f4b1a92ed6c13e0cd21534df56155cf3ef71f87"
fi
