# tester Build_figures_for_articles:

#### lunching Notebook :
python3 -m pip install --user jupyter

python3 -m notebook ATT-main-10/wacv_reporduce_graphs

#### Figure 10.b : Frobenius norm of observation error ||A - ∇H^{-1}_{y_i} H_F||_F
run build_figures_for_articles.py

#### Figure 11 : Error between deformed pics by tilt and extracted pics from autocorrelation
run graph_tilt_error.py

#### Figure 9.c : Occlusion graph
run generate_dataset.py
run run_occlusion.py
