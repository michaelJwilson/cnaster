MISCELLANEOUS
- 20% of SNPs are "lost" as their position does not overlap with the gencode 38 reference list;
- initial clone assigned assuming common meaning between position across slices.
- ln 216 of hmm sitewise: efficient eval. of BAF emission prob. for phased state given tumor proportion.
- Breakdown of snps per block: include number snp-covering umis.
- sample HT112C1-U1,HT112C1-U2 gives 18 min. runtime.


RECOMB/ISMB
- abstract submission deadline: Nov 7 2025; 34 days.
- full paper submission deadline: Nov 14 2025; 41 days.


KEY
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


TODAY
- edge dependent p_add.

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