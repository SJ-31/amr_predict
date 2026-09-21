#!/usr/bin/env bash


pushd ../data/remote/blast/ || exit

blast_fmt () {
    blastdbcmd -db "${1}/${1}" -entry all | seqkit seq -o "${1}.fq.gz"
    blastdbcmd -db "${1}/${1}" -entry all -outfmt "%a %T" >> taxidmapping.txt
    
}

blast_fmt 16S_ribosomal_RNA
blast_fmt 28S_fungal_sequences
blast_fmt ITS_eukaryote_sequences
blast_fmt refseq_rna

seqkit seq *fq.gz -o db/tmp.fq.gz
rm *fq.gz

mmseqs createdb db/tmp.fq.gz db/nt
mmseqs createtaxdb db/nt tmp --ncbi-tax-dump taxonomy/ --tax-mapping-file \
    taxidmapping.txt

rm db/tmp.fq.gz

popd || exit
