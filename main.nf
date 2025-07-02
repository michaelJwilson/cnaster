#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.sample_list = "./mock/mock_sample_list.tsv"
params.outputdir = "./results"
params.container = "cnaster.sif"
params.config = "config.yaml"

// TODO e.g. derive from config.
params.eagle_dir = "/u/congma/ragr-data/users/congma/environments/Eagle_v2.4.1/"
params.phasing_panel = "/u/congma/ragr-data/users/congma/references/phasing_ref/1000G_hg38/"
params.ref_snp_vcf = "/u/congma/ragr-data/users/congma/references/snplist/nocpg.genome1K.phase3.SNP_AF5e4.chr1toX.hg38.vcf.gz"
params.hgtable = "/u/congma/ragr-data/users/congma/Codes/STARCH_crazydev/hgTables_hg38_gencode.txt"
params.geneticmap = "<REPLACE_ME>"
params.contigs = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22"

process cellsnp_lite_pileup {
    tag "${sample_id}"
    container params.container
    cpus 4
    memory '8 GB'
    
    publishDir "${params.outputdir}/pileup", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bam), path(bai), path(barcodes), path(ref_snp_vcf)
    
    output:
    tuple val(sample_id), path("${sample_id}"), emit: pileup_dir
    path "${sample_id}_cellsnp.log", emit: log
    
    script:
    """
    echo "Running cellsnp-lite for sample: ${sample_id}" > ${sample_id}_cellsnp.log
    echo "Input BAM: ${bam}" >> ${sample_id}_cellsnp.log
    echo "Barcodes file: ${barcodes}" >> ${sample_id}_cellsnp.log
    echo "Region VCF: ${ref_snp_vcf}" >> ${sample_id}_cellsnp.log
    echo "Threads: ${task.cpus}" >> ${sample_id}_cellsnp.log
    
    cellsnp-lite -s ${bam} \\
                 -b ${barcodes} \\
                 -O ${sample_id}/pileup/ \\
                 -R ${ref_snp_vcf} \\
                 -p ${task.cpus} \\
                 --minMAF 0 \\
                 --minCOUNT 2 \\
                 --UMItag Auto \\
                 --cellTAG CB \\
                 --gzip
                 >> ${sample_id}_cellsnp.log 2>&1
    
    echo "cellsnp-lite completed for ${sample_id}" >> ${sample_id}_cellsnp.log
    """
}


process filter_snps_forphasing {
    tag "${sample_id}"
    container params.container
    cpus 2
    memory '4 GB'
    
    publishDir "${params.outputdir}/filtered_snps", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    
    output:
    tuple val(sample_id), path("${sample_id}_filtered_snps.vcf.gz"), emit: filtered_snps
    path "${sample_id}_filter_snps.log", emit: log
    
    script:
    """
    echo "Filtering SNPs for phasing: ${sample_id}" > ${sample_id}_filter_snps.log
    echo "Input BAM: ${bam}" >> ${sample_id}_filter_snps.log
    echo "Output directory: \$(pwd)" >> ${sample_id}_filter_snps.log
    
    prep_snps \\
        ${sample_id} \\
        \$(pwd) \\
        >> ${sample_id}_filter_snps.log 2>&1
    
    if [ -f "${sample_id}_filtered_snps.vcf" ]; then
        bgzip "${sample_id}_filtered_snps.vcf"
        tabix -p vcf "${sample_id}_filtered_snps.vcf.gz"
    elif [ ! -f "${sample_id}_filtered_snps.vcf.gz" ]; then
        echo "ERROR: Expected output file not found" >> ${sample_id}_filter_snps.log
        exit 1
    fi
    
    echo "SNP filtering completed for ${sample_id}" >> ${sample_id}_filter_snps.log
    """
}

process eagle2_phasing {
    tag "chr${chrname}"
    container params.container
    cpus 4
    memory '8 GB'
    
    publishDir "${params.outputdir}/eagle2_phasing", mode: 'copy'
    
    input:
    tuple val(chrname), path(vcf)
    
    output:
    tuple val(chrname), path("chr${chrname}.phased.vcf.gz"), emit: phased_vcf
    path "phasing_chr${chrname}.log", emit: log
    
    script:
    """
    echo "Starting phasing for contig ${chrname}" > phasing_chr${chrname}.log
    
    ref_panel="${params.phasing_panel}/chr${chrname}.genotypes.bcf"

    if [ ! -f "\$ref_panel" ]; then
        echo "Reference panel not found: \$ref_panel" >> phasing_chr${chrname}.log
        echo "Skipping phasing, copying input VCF" >> phasing_chr${chrname}.log
        cp ${vcf} chr${chrname}.phased.vcf.gz
        exit 0
    fi
    
    eagle \\
        --numThreads ${task.cpus} \\
        --vcfTarget ${vcf} \\
        --vcfRef "\$ref_panel" \\
        --geneticMapFile ${params.genetic_map} \\
        --outPrefix chr${chrname}.phased \\
        >> phasing_chr${chrname}.log 2>&1
    
    echo "Phasing completed for contig ${chrname}" >> phasing_chr${chrname}.log
    """
}

process create_allele_matrices {
    tag "${sample_id}"
    container params.container
    cpus 2
    memory '8 GB'
    
    publishDir "${params.outputdir}/allele_matrices", mode: 'copy'
    
    input:
    tuple val(sample_id), path(cellsnp_dir), path(eagle_dir), path(barcode_file)
    
    output:
    tuple val(sample_id), path("${sample_id}_cell_snp_Aallele.npz"), emit: a_allele
    tuple val(sample_id), path("${sample_id}_cell_snp_Ballele.npz"), emit: b_allele
    tuple val(sample_id), path("${sample_id}_unique_snp_ids.npy"), emit: snp_ids
    path "${sample_id}_allele_matrices.log", emit: log
    
    script:
    """
    echo "Creating allele matrices for sample: ${sample_id}" > ${sample_id}_allele_matrices.log
    echo "cellsnp-lite results: ${cellsnp_dir}" >> ${sample_id}_allele_matrices.log
    echo "Eagle results: ${eagle_dir}" >> ${sample_id}_allele_matrices.log
    echo "Barcode file: ${barcode_file}" >> ${sample_id}_allele_matrices.log
    
    create_allele_matrices \\
        -c ${cellsnp_dir} \\
        -e ${eagle_dir} \\
        -b ${barcode_file} \\
        -o ${sample_id}_matrices \\
        >> ${sample_id}_allele_matrices.log 2>&1
    
    mv ${sample_id}_matrices/cell_snp_Aallele.npz ${sample_id}_cell_snp_Aallele.npz
    mv ${sample_id}_matrices/cell_snp_Ballele.npz ${sample_id}_cell_snp_Ballele.npz
    mv ${sample_id}_matrices/unique_snp_ids.npy ${sample_id}_unique_snp_ids.npy
    
    echo "Allele matrices creation completed for ${sample_id}" >> ${sample_id}_allele_matrices.log
    """
}

// (sudo) nextflow run main.nf --sample_list ./mock/mock_sample_list.tsv --outputdir ./results --container cnaster.sif --config config.yaml
workflow {
    sample_ch = Channel
        .fromPath(params.sample_list)
        .splitCsv(header: true, sep: ' ')
        .map { row -> 
            tuple(
                row.sample_id,
                file(row.bam),
                file(row.bai),
                file(row.barcodes)
            )
        }

    ref_snp_vcf_ch = Channel.fromPath(params.ref_snp_vcf)
    chr_ch = Channel.from(params.contigs.split(','))

    sample_ids = sample_ch
        .map { sample_id, bam, bai -> sample_id }
        .collect()
    
    sample_ids.subscribe { ids ->
        println """
        =====================================
        CNAster Pipeline Run
        =====================================
        Samples to process: ${ids.size()}
        Sample IDs: ${ids.join(', ')}
        
        Contig range: chr${params.contigs.split(',').first()} - chr${params.contigs.split(',').last()}
        
        Output directory: ${params.outputdir}
        Container: ${params.container}
        =====================================
        """
    }

    cellsnp_lite_pileup(
        sample_ch.combine(ref_snp_vcf_ch)
    )
    
    // ....
}