import numpy as np

def perturb_phase(cell_snp_Aallele, cell_snp_Ballele):
    n_snps = cell_snp_Aallele.shape[1]

    # Generate random switches: True means a switch occurs at this SNP
    switches = np.random.random(n_snps) < switch_rate

    # Determine state: 0 = normal, 1 = switched
    # cumsum % 2 creates runs of switched/unswitched states
    states = np.cumsum(switches) % 2

    # Identify indices where the phase is switched
    swapped_indices = np.where(states == 1)[0]

    # Create copies to avoid modifying originals
    new_A = cell_snp_Aallele.copy()
    new_B = cell_snp_Ballele.copy()

    # Swap counts for the switched SNPs
    new_A[:, swapped_indices] = cell_snp_Ballele[:, swapped_indices]
    new_B[:, swapped_indices] = cell_snp_Aallele[:, swapped_indices]

    return new_A, new_B
