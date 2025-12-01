#!/bin/bash
set -o pipefail

ROOT="/Users/mw9568/Work/ragr/sim"

# TODO CHECK
# numcnas1.2_cnasize1e7_ploidy2_random3   1
# numcnas6.3_cnasize5e7_ploidy2_random6   0
# numcnas1.2_cnasize3e7_ploidy2_random7   1
# numcnas6.3_cnasize3e7_ploidy2_random4   3
# numcnas3.3_cnasize3e7_ploidy2_random0   1
# numcnas1.2_cnasize1e7_ploidy2_random6   1

RANDOM_STATES=(1)
SAMPLE_IDS=("numcnas3.3_cnasize3e7_ploidy2_random0")

NUM_CLONES=3

for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
    SAMPLE_DIR="${ROOT}/nomixing_cnaster_related/${SAMPLE_ID}"
    
    TRUE_PLOTS_DIR="${SAMPLE_DIR}"
    echo "Ground truth: ${TRUE_PLOTS_DIR}"
    
    open "${TRUE_PLOTS_DIR}/true_acn_profile.pdf"
    open "${TRUE_PLOTS_DIR}/true_clones_spatial.pdf"
    
    for RANDOM_STATE in "${RANDOM_STATES[@]}"; do
        EST_PLOTS_DIR="${SAMPLE_DIR}/clone${NUM_CLONES}_rectangle${RANDOM_STATE}_w1.0/plots"
        echo "Estimated:    ${EST_PLOTS_DIR}"
        
        open "${EST_PLOTS_DIR}/clones_genomic.pdf"
        open "${EST_PLOTS_DIR}/clones_spatial.pdf"
        open "${EST_PLOTS_DIR}/${SAMPLE_ID}_rectangle${RANDOM_STATE}_copy_states.pdf"
        open "${EST_PLOTS_DIR}/initial_clones_spatial.pdf"
        open "${EST_PLOTS_DIR}/bafonly_clones_spatial.pdf"
        open "${EST_PLOTS_DIR}/merged_bafonly_clones_spatial.pdf"
        
    done
done
