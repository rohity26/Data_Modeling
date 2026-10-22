"""build_newdata_tcp.py — Build test_newdata.npz for the TCP-traffic collection
in data_input/test_csv_tcp/, writing into a SEPARATE output tree
(CU_NET_TCP_STRESS/) so it never touches CU_NET_random_STRESS/, which
run_experiment_newdata.py may be reading/writing concurrently.

Same logic as build_newdata_test.py, just repointed. Only topologies whose
CU + all DUs are fully collected are built — as of this run, that's
cu0_du0du1 (needs du0, du1) and cu1_du2 (needs du2). cu2_du3du4du5 and
cu3_du6du7du8du9 are skipped until their remaining DUs finish collecting.

train.npz for each topology/stress folder under CU_NET_TCP_STRESS/ is a
plain copy of the corresponding CU_NET_random_STRESS/.../train.npz (same
normal training data, protocol-agnostic — copied once, not regenerated here).

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python build_newdata_tcp.py
"""

from __future__ import annotations

from pathlib import Path

import build_dataset as bd
import build_newdata_test as bnt

bnt.NEW_CSV_DIR = Path("data_input/test_csv_tcp")
bnt.BASE_DIR = Path("CU_NET_TCP_STRESS")

READY_TOPOLOGIES = ["cu0_du0du1", "cu1_du2"]  # update as more DUs finish collecting


def main():
    for stress_type in bnt.STRESS_TYPES:
        for topology in READY_TOPOLOGIES:
            bnt.build_one(topology, stress_type)


if __name__ == "__main__":
    main()
