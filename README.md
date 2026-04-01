# lunching Notebook :
python3 -m pip install --user jupyter

projections manipulation : python3 -m notebook WACV2026-Autocorrelation-based-Fiducial-Markers-for-Traceability-main/test_projections
graph visualisation : python3 -m notebook WACV2026-Autocorrelation-based-Fiducial-Markers-for-Traceability-main/wacv_reporduce_graphs
# test Build_figures_for_articles:
#### Figure 10.b : Frobenius norm of observation error ||A - ∇H^{-1}_{y_i} H_F||_F
run build_figures_for_articles.py

#### Figure 11 : Error between deformed pics by tilt and extracted pics from autocorrelation
run graph_tilt_error.py

#### Figure 9.c : Occlusion graph
run generate_dataset.py
run run_occlusion.py
