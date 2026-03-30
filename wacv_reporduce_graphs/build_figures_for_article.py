import numpy as np
from graph_Ai_Jc_energy import heatmap_affinity_error_parallel 
from ghostseal_generator import generate_gs3d_noise_deformation


#_____Params_____
start_ps = 120
min_ps = 60
max_ps = 200
step = 5
tol_abs = 1
tol_rel_pct = 1.0
ref_smooth_window = 4
stable_seq_len = 4
show_tracks = False

hex_kwargs = dict(
    k=20,
    nms_size=50,
    exclude_center_radius=20.0,
    min_separation=20,
    refine_model='tps',
    refine_halfwin=1.5,
    tps_coarse_step=0.25,
    energy_halfwin=1.5,
    min_dist=20,
    antipodal_tol=10.0,
    angle_min_deg=10.0,
    w_exclude_center_radius=20.0,
)


if __name__ == "__main__":
    # Figure 10.b : Frobenius norm of observation error ||A - ∇H^{-1}_{y_i} H_F||_F
    large_noise, gs3d1, matrix = generate_gs3d_noise_deformation(
        2000, 2000, (60, 0), (0, 60),
        1, 0, 1, 0, 20, 10, seed=21353456
    )

    E, ax_list, ay_list = heatmap_affinity_error_parallel(
        U_ref=(60, 0), V_ref=(0, 60),
        image_size=2000,
        ax_list=np.linspace(-40, 40, 120),
        ay_list=np.linspace(-40, 40, 120),
        hex_kwargs=hex_kwargs,
        start_ps=start_ps, min_ps=min_ps, max_ps=max_ps, step=step,
        tol_abs=tol_abs, tol_rel_pct=tol_rel_pct,
        ref_smooth_window=ref_smooth_window, stable_seq_len=stable_seq_len,
        which_pair="all",
        show=True,
        json_out_path="affinity_error_grid.json",
        show_tracks=False,     
        max_workers=None      
    )
