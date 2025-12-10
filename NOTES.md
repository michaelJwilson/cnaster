MISCELLANEOUS
- 20% of SNPs are "lost" as their position does not overlap with the gencode 38 reference list;
- initial clone assigned assuming common meaning between position across slices.
- ln 216 of hmm sitewise: efficient eval. of BAF emission prob. for phased state given tumor proportion.
- Breakdown of snps per block: include number snp-covering umis.
- sample HT112C1-U1,HT112C1-U2 gives 18 min. runtime.


RECOMB/ISMB
- abstract submission deadline: Nov 7 2025; 34 days.
- full paper submission deadline: Nov 14 2025; 41 days.

TODAY
- merge_by_minspots spatial.

COMMANDS
- ps aux | grep batch_zenodo_sim_parallel_run.sh
- caffeinate -d -w <PID>

RUNTIME
- omics.py:581 summarize_counts_for_blocks
- spatial.py:306 sufficient_umis_clone_assignment
- hmm_initialize.py:181

KEY
- betabinomial likelihood at p=0., 1.
- Clone assignment swaps that re-calculates new parameters.
- propagate BAF values from baf only run to RDR refinement.
- transcript definition based on visium gene selection vs gencode coding regions!
- +522 omics.py enforce secondary_min_umi for last block.
- Deprecate Neyman-Pearson.
- normal tumor prop. from LOH - sequencing error? 
- filter on X umi count for choosing initial states.
- +579 hmrf.py feed clone_stack vs not to GMM.
- ln mu -> mu convergence check.
- added jitter to NB start. turned up disp.
- new non-local sampling: annealed Wolff?
- model selection in spatial correlation.
- copy mixture initialization.
- clone initialization.


QUESTIONS
- Poor initial copy state parameters can dominate likelihood over spatial coherence?
- ICM is optimal for spatial_weight=0; ICM converges in 1 suggests spatial_weight is negligible:
  spatial_weight is too low!  Coherence implied by intertia (last clone proportion); clone initialization &
  aggregated pseudobulk; ICM as greedy max.


TODO major
- vanilla CalicoST run & results.
- simulation based tests.
- tumor proportion emission.
- p_add given temperature.
- em convergence for rdr? data points well separated by state ...
- 


TODO minor
- adjacency matrix is symmetric or zeroed??  see icm.py
- HT306P1-U1 has inconsistent barcodes between snps and umis (ahhhh!); motivation: single slice.
- plot of edge/pooling definition: latter reduces computation requirement.
- check & test logsumexp implementation for icm_sweep.
- independent plotting run; copy mixture seeding plot.
- independent validation of cost.
- clone labels on rdr/baf plot.
- copy state filtering on rdr/baf plot.


ANALYSIS
- clone initialization results
- wolff algorithm results.


READING
- quantum until Wed. 8th; qiskit (rust based).
- loopy belief propagation
- k-means, vi, gmm, in McKay.
- hmm in Durbin.
- graph cuts in Kleinberg & Tardos.