# Human In the Loop Project - Towards Fair Link Prediction using Active Learning

Installation can be done by running the following command on Snellius Cluster:
```bash
sbatch install_environment.job
```
Find all the generated graphs - with and without active learning here: https://amsuni-my.sharepoint.com/:f:/g/personal/madhura_pawar_student_uva_nl/EvDBQmphSeRLgVoeAP8uzPkB9luqqz1HKFWMAFvSWDTQ5Q?e=5SyjAP

(Feel free to request access, if need files!)

```--no_human``` flag can be added to run an active learning loop and visualization of the community.


Command to run active learning loop:
```bash
sbatch run_job.job
```

Command to generate results:
```bash
python3 generate_results.py --fm ____  --hMM ____  --hmm ____ 
```

Command to generate plot for visualization of dispersion of minorities in the community:
```bash
python3 visualize_community.py --fm ____ --hMM ____ --hmm ____
```
