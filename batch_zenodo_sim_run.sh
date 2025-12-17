#!/bin/bash
set -o pipefail

# ROOT="/u/mw9568/scratch/calicost_sims/"
ROOT="/Users/mw9568/Work/ragr/sim/"

SEED=12345
RANDOM_STATES=(0 1 2 3 4) 
USE_EXISTING=true

SAMPLE_IDS=()

for d in "$ROOT"/"simulated_data_related"/*/; do
   SAMPLE_IDS+=("$(basename "$d")")
done

SAMPLE_IDS=($(printf "%s\n" "${SAMPLE_IDS[@]}" | gshuf --random-source=<(yes $SEED)))

# DEBUGGING
USE_EXISTING=false
# SAMPLE_IDS=("numcnas1.2_cnasize1e7_ploidy2_random0")
# SAMPLE_IDS=("numcnas3.3_cnasize3e7_ploidy2_random0")

# RANDOM_STATES=(4) 
# SAMPLE_IDS=("numcnas3.3_cnasize5e7_ploidy2_random1") # MIN SPOTS per clone=100

# RANDOM_STATES=(3)
# SAMPLE_IDS=("numcnas3.3_cnasize5e7_ploidy2_random6")

# RANDOM_STATES=(3)
# SAMPLE_IDS=("numcnas6.3_cnasize3e7_ploidy2_random4")

# RANDOM_STATES=(0)
SAMPLE_IDS=("numcnas6.3_cnasize5e7_ploidy2_random6")

# echo "${SAMPLE_IDS[@]}"
echo "Found ${#SAMPLE_IDS[@]} sample ids @ ${ROOT}"
echo "Running with ${#RANDOM_STATES[@]} random states: ${RANDOM_STATES[@]}"

# mkdir -p logs errors zenodo_sample_sheets

for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
    OUT_BASE="${ROOT}/nomixing_cnaster_related/${SAMPLE_ID}"

    # NB replace numcnas1.2_cnasize1e7_ploidy2_random0 in ./zenodo_sample_sheet.tsv with {SAMPLE_ID} and write to zenodo_sample_sheets/zenodo_{SAMPLE_ID}_sheet.tsv                                                                                                          
    sed "s|numcnas1.2_cnasize1e7_ploidy2_random0|${SAMPLE_ID}|g; s|Z001-U1|${SAMPLE_ID}|g" ./zenodo_sample_sheet.tsv > "zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv"
    
    for RANDOM_STATE in "${RANDOM_STATES[@]}"; do
        OUT_PATTERN="${OUT_BASE}/clone?_rectangle${RANDOM_STATE}_w1.0/rdrbaf_final_nstates?_smp.npz"

        if [[ "$USE_EXISTING" == "true" ]]; then
            if compgen -G "$OUT_PATTERN" > /dev/null; then
                echo "Utilizing existing results for SAMPLE_ID=${SAMPLE_ID}; RANDOM_STATE=${RANDOM_STATE}."
                continue
            fi
        fi

        LOG_PATH="logs/cnaster_${SAMPLE_ID}_${RANDOM_STATE}.log"
        PERF_PATH="logs/cnaster_${SAMPLE_ID}_${RANDOM_STATE}.perf"
        
        rm -f "${LOG_PATH}"
        rm -f "${PERF_FILE}"      
        
        run_cnaster zenodo_sim_config.yaml \
            -o "hmrf.random_state=${RANDOM_STATE}" \
            -o "paths.sample_sheet=zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv" \
            -o "paths.output_dir=${ROOT}/nomixing_cnaster_related/${SAMPLE_ID}/" \
            -o "paths.perf_path=${PERF_PATH}" \
            2>&1 | tee "${LOG_PATH}"

        echo "Solved for SAMPLE_ID=${SAMPLE_ID}, RANDOM_STATE=${RANDOM_STATE} to ${LOG_PATH} and ${PERF_PATH}"
        
        rc=${PIPESTATUS[0]}
        
        if [[ $rc -ne 0 ]]; then
            echo "run_cnaster failed for SAMPLE_ID=${SAMPLE_ID}, RANDOM_STATE=${RANDOM_STATE} (rc=${rc})" >&2
            mv "logs/cnaster_${SAMPLE_ID}_${RANDOM_STATE}.log" "errors/cnaster_${SAMPLE_ID}_${RANDOM_STATE}.err"
        fi
    done    
done
