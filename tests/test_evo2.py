#!/usr/bin/env python

from subprocess import run

script = "/data/home/shannc/amr_predict/scripts/evo2_sbatch.sh"
input = "/data/home/shannc/amr_predict/tests/data/evo2/one.fasta"
outdir = "/data/home/shannc/amr_predict/tests/data/evo2/from_py"
evo_run = run(
    f"sbatch --wait --parsable {script} -i {input} -o {outdir}",
    shell=True,
    capture_output=True,
)
print(evo_run.stdout.decode())
