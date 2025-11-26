#!/bin/bash
set -o pipefail

# ROOT="/u/mw9568/scratch/calicost_sims/"
ROOT="/Users/mw9568/Work/ragr/sim/"

SEED=12345
NUM_STARTS=5
USE_EXISTING=true

# SAMPLE_IDS=("numcnas1.2_cnasize1e7_ploidy2_random0")
# SAMPLE_IDS=("numcnas3.3_cnasize3e7_ploidy2_random0")

rm -f cnaster.log
rm -f cnaster.perf

SAMPLE_IDS=()

for d in "$ROOT"/"simulated_data_related"/*/; do
   SAMPLE_IDS+=("$(basename "$d")")
done

SAMPLE_IDS=($(printf "%s\n" "${SAMPLE_IDS[@]}" | gshuf --random-source=<(yes $SEED)))

# echo "${SAMPLE_IDS[@]}"
echo "Found ${#SAMPLE_IDS[@]} sample ids @ ${ROOT}"

# mkdir -p logs errors zenodo_sample_sheets

for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
    OUT_BASE="${ROOT}/nomixing_cnaster_related/${SAMPLE_ID}"

    # NB replace numcnas1.2_cnasize1e7_ploidy2_random0 in ./zenodo_sample_sheet.tsv with {SAMPLE_ID} and write to zenodo_sample_sheets/zenodo_{SAMPLE_ID}_sheet.tsv                                                                                                          
    sed "s|numcnas1.2_cnasize1e7_ploidy2_random0|${SAMPLE_ID}|g; s|Z001-U1|${SAMPLE_ID}|g" ./zenodo_sample_sheet.tsv > "zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv"
    
    for ((RANDOM_STATE=0; RANDOM_STATE<NUM_STARTS; RANDOM_STATE++)); do
        echo "Solving for SAMPLE_ID=${SAMPLE_ID}, RANDOM_STATE=${RANDOM_STATE}"

        OUT_PATTERN="${OUT_BASE}/clone?_rectangle${RANDOM_STATE}_w1.0/rdrbaf_final_nstates?_smp.npz"

        if [[ "$USE_EXISTING" == "true" ]]; then
            if compgen -G "$OUT_PATTERN" > /dev/null; then
                echo "Utilizing existing results for SAMPLE_ID=${SAMPLE_ID}; RANDOM_STATE=${RANDOM_STATE}."
                continue
            fi
        fi

        run_cnaster zenodo_sim_config.yaml \
            -o "hmrf.random_state=${RANDOM_STATE}" \
            -o "paths.sample_sheet=zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv" \
            -o "paths.output_dir=${ROOT}/nomixing_cnaster_related/${SAMPLE_ID}/" \
            2>&1 | tee "logs/cnaster_${SAMPLE_ID}_${RANDOM_STATE}.log"
        
        rc=${PIPESTATUS[0]}
        
        if [[ $rc -ne 0 ]]; then
            echo "run_cnaster failed for SAMPLE_ID=${SAMPLE_ID}, RANDOM_STATE=${RANDOM_STATE} (rc=${rc})" >&2
            mv "logs/cnaster_${SAMPLE_ID}_${RANDOM_STATE}.log" "errors/cnaster_${SAMPLE_ID}_${RANDOM_STATE}.err"
        fi
    done    
done
