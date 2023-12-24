# Human In the Loop Project - Towards Fair Link Prediction using Active Learning

Installation can be done by running the following command on Snellius Cluster:
```bash
sbatch install_environment.job
```
Find all the generated graphs - with and without active learning here

Command to run active learning loop:
```bash
sbatch run_job.job
```

Command to generate results:
```bash
python3 generate_results.py --fm << >> --hMM << >> --hmm << >>
```

Command to generate plot for visualization of dispersion of minorities in the community:
```bash
python3 visualize_community.py --fm << >> --hMM << >> --hmm << >>
```
