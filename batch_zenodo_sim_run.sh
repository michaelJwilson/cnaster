#!/bin/bash
set -o pipefail

# ROOT="/u/mw9568/scratch/calicost_sims/"
ROOT="/Users/mw9568/Work/ragr/sim/"

NUM_STARTS=5

# SAMPLE_IDS=("numcnas1.2_cnasize1e7_ploidy2_random0")
SAMPLE_IDS=("numcnas3.3_cnasize3e7_ploidy2_random0")

rm -f cnaster.log
rm -f cnaster.perf

for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
    # NB replace numcnas1.2_cnasize1e7_ploidy2_random0 in ./zenodo_sample_sheet.tsv with {SAMPLE_ID} and write to zenodo_sample_sheets/zenodo_{SAMPLE_ID}_sheet.tsv                                                                                                          
    sed "s|numcnas1.2_cnasize1e7_ploidy2_random0|${SAMPLE_ID}|g; s|Z001-U1|${SAMPLE_ID}|g" ./zenodo_sample_sheet.tsv > "zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv"
    
    for ((RANDOM_STATE=0; RANDOM_STATE<NUM_STARTS; RANDOM_STATE++)); do
        echo "Solving for SAMPLE_ID=${SAMPLE_ID}, RANDOM_STATE=${RANDOM_STATE}"

        if ! run_cnaster zenodo_sim_config.yaml \
            -o "hmrf.random_state=${RANDOM_STATE}" \
            -o "paths.sample_sheet=zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv" \
            -o "paths.output_dir=${ROOT}/nomixing_cnaster_related/${SAMPLE_ID}/" \
            # -o "annotation.clone_label=${ROOT}/simulated_data_related/${SAMPLE_ID}/truth_clone_labels.tsv" \
            # -o "annotation.true_cnv=${ROOT}/simulated_data_related/${SAMPLE_ID}/truth_acn_profile.tsv"
        then
            rc=$?
            echo "run_cnaster failed for SAMPLE_ID=${SAMPLE_ID}, RANDOM_STATE=${RANDOM_STATE} (rc=${rc})" >&2
            exit 1  # exits inner RANDOM_STATE loop; use 'break 2' to exit both loops, or 'exit 1' to stop script
        fi
    done
done
