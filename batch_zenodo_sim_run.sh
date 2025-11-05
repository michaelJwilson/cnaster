#!/bin/bash
SAMPLE_IDS=("numcnas1.2_cnasize1e7_ploidy2_random0")

for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
    echo "Solving for ${SAMPLE_ID}"
    
    # NB replace numcnas1.2_cnasize1e7_ploidy2_random0 in ./zenodo_sample_sheet.tsv with {SAMPLE_ID} and write to zenodo_sample_sheets/zenodo_{SAMPLE_ID}_sheet.tsv
    sed "s|numcnas1.2_cnasize1e7_ploidy2_random0|${SAMPLE_ID}|g" ./zenodo_sample_sheet.tsv > "zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv"
    
    run_cnaster zenodo_sim_config.yaml \
         -o "paths.sample_sheet=zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv" \
         -o "paths.output_dir=/u/mw9568/scratch/calicost_sims/nomixing_cnaster_related/${SAMPLE_ID}/" \
         -o "annotation.clone_label=/u/mw9568/scratch/calicost_sims/simulated_data_related/${SAMPLE_ID}/truth_clone_labels.tsv"
done
