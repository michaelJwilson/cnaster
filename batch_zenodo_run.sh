#!/bin/bash
set -o pipefail

# ROOT="/u/mw9568/scratch/calicost_sims/"
ROOT="/Users/mw9568/Work/ragr/sim/"

SEED=12345
RANDOM_STATES=(0 1 2 3 4)
USE_EXISTING=false
MAX_JOBS=1

echo "Preparing workspace..."
mkdir -p logs errors zenodo_sample_sheets

SAMPLE_IDS=()
for d in "$ROOT"/"simulated_data_related"/*/; do
   SAMPLE_IDS+=("$(basename "$d")")
done

SAMPLE_IDS=($(printf "%s\n" "${SAMPLE_IDS[@]}" | gshuf --random-source=<(yes $SEED)))

# --- DEBUGGING OVERRIDES ---
# SAMPLE_IDS=("numcnas1.2_cnasize1e7_ploidy2_random0")                                                                                                                                                               
# SAMPLE_IDS=("numcnas3.3_cnasize3e7_ploidy2_random0")                                                                                                                                                                 
# RANDOM_STATES=(4)                                                                                                                                                                                                   
# SAMPLE_IDS=("numcnas3.3_cnasize5e7_ploidy2_random1") # MIN SPOTS per clone=100

# RANDOM_STATES=(3)                                                                                                                                                                                                   
# SAMPLE_IDS=("numcnas3.3_cnasize5e7_ploidy2_random6")

# RANDOM_STATES=(3)                                                                                                                                                                                                   
# SAMPLE_IDS=("numcnas6.3_cnasize3e7_ploidy2_random4")                                                                                                                                                                 
RANDOM_STATES=(0)
SAMPLE_IDS=("numcnas6.3_cnasize5e7_ploidy2_random6")

echo "Found ${#SAMPLE_IDS[@]} sample ids @ ${ROOT}"
echo "Running with ${#RANDOM_STATES[@]} random states: ${RANDOM_STATES[@]}"

for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
    sed "s|numcnas1.2_cnasize1e7_ploidy2_random0|${SAMPLE_ID}|g; s|Z001-U1|${SAMPLE_ID}|g" \
        ./zenodo_sample_sheet.tsv > "zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv"
done

echo "Sample sheets generated."

run_single_job() {
    local SAMPLE_ID=$1
    local RANDOM_STATE=$2
    local ROOT=$3
    local USE_EXISTING=$4
    
    local OUT_BASE="${ROOT}/nomixing_cnaster_related/${SAMPLE_ID}"
    local LOG_PATH="logs/cnaster_${SAMPLE_ID}_${RANDOM_STATE}.log"
    local PERF_PATH="logs/cnaster_${SAMPLE_ID}_${RANDOM_STATE}.perf"

    if [[ "$USE_EXISTING" == "true" ]]; then
        shopt -s nullglob
        local existing_files=(${OUT_BASE}/clone?_rectangle${RANDOM_STATE}_w1.0/rdrbaf_final_nstates?_smp.npz)
        shopt -u nullglob
        
        if [[ ${#existing_files[@]} -gt 0 ]]; then
            # -mtime -1 accurately checks for modifications within the last 24 hours
            if find "${existing_files[@]}" -maxdepth 0 -mtime -1 2>/dev/null | grep -q .; then
                echo "Utilizing existing results (modified < 24h) for SAMPLE_ID=${SAMPLE_ID}; RANDOM_STATE=${RANDOM_STATE}."
                return 0
            fi
        fi
    fi
    
    rm -f "${LOG_PATH}" "${PERF_PATH}"

    echo "Solving for SAMPLE_ID=${SAMPLE_ID}, RANDOM_STATE=${RANDOM_STATE} to ${LOG_PATH} and ${PERF_PATH}"

    run_cnaster zenodo_sim_config.yaml \
        -o "hmrf.random_state=${RANDOM_STATE}" \
        -o "paths.sample_sheet=zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv" \
        -o "paths.output_dir=${OUT_BASE}/" \
        -o "paths.perf_path=${PERF_PATH}" \
        2>&1 | tee "$LOG_PATH"
     
    local rc=${PIPESTATUS[0]}
    
    if [[ $rc -ne 0 ]]; then
        echo "run_cnaster failed for SAMPLE_ID=${SAMPLE_ID}, RANDOM_STATE=${RANDOM_STATE} (rc=${rc})" >&2
        mv "$LOG_PATH" "errors/cnaster_${SAMPLE_ID}_${RANDOM_STATE}.err"
        return $rc
    fi
    
    return 0
}

# Export function and variables so GNU parallel can access them inside subshells
export -f run_single_job
export ROOT USE_EXISTING

JOBS=()
for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
    for RANDOM_STATE in "${RANDOM_STATES[@]}"; do
        JOBS+=("$SAMPLE_ID $RANDOM_STATE")
    done
done

printf "%s\n" "${JOBS[@]}" | parallel -j "$MAX_JOBS" --colsep ' ' \
    run_single_job {1} {2} "$ROOT" "$USE_EXISTING"

echo "All cnaster jobs completed."
