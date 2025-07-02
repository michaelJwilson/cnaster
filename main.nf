#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.greeting = "Hello"
params.name = "World"
params.outdir = "./results"

process sayHello {
    tag "greeting"
    publishDir params.outdir, mode: 'copy'
    
    input:
    val greeting
    val name
    
    output:
    path "greeting.txt"
    
    script:
    """
    echo "${greeting}, ${name}!" > greeting.txt
    echo "This is Nextflow DSL2" >> greeting.txt
    echo "Process completed at: \$(date)" >> greeting.txt
    """
}

workflow {
    greeting_ch = Channel.of(params.greeting)
    name_ch = Channel.of(params.name)
    
    sayHello(greeting_ch, name_ch)
    
    sayHello.out.view { "Created file: $it" }
}

workflow.onComplete {
    println "Hello World workflow completed!"
    println "Results saved to: ${params.outdir}"
}