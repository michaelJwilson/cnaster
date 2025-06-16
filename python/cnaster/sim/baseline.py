import numpy as np

def generate_baseline(config):
    num_segments = config.mappable_genome_kbp // config.segment_size_kbp
    exp_gene_segment = config.exp_gene_kbp * config.segment_size_kbp
    
    lambdas = np.random.poisson(lam=exp_gene_segment, size=num_segments)
    lambdas /= lambdas.sum()

    return lambdas