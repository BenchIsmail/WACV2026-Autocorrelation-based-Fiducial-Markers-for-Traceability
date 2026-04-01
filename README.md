# lunching Notebook :

python3 -m pip install --user jupyter

in terminal (Mac) cmd(windows)

go to folder WACV2026-Autocorrelation-based-Fiducial-Markers-for-Traceability-main/wacv_reporduce_graphs

##### For projections manipulation : 

python3 -m notebook test_projections.ipynb

##### For graph visualisation : 

python3 -m notebook notebook_experiments.ipynb

# test Build_figures_for_articles:
#### Figure 10.b : Frobenius norm of observation error ||A - ∇H^{-1}_{y_i} H_F||_F
run build_figures_for_articles.py

#### Figure 11 : Error between deformed pics by tilt and extracted pics from autocorrelation
run graph_tilt_error.py

#### Figure 9.c : Occlusion graph
run generate_dataset.py
run run_occlusion.py
