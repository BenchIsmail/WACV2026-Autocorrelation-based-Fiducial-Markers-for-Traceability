import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

import rectification as rectif
from qr_code import estimate_qrcode_homography




def erreur_frobenius(M1: np.ndarray, M2: np.ndarray):
    errs = [np.linalg.norm(M1[:3,:2] - M2[:3,:2], ord='fro')]
    return errs



def _ensure_metadata_index(metadata_or_path):
    if isinstance(metadata_or_path, str):
        with open(metadata_or_path, "r") as f:
            data = json.load(f)
    else:
        data = metadata_or_path

    if isinstance(data, dict):
        candidates = []
        for v in data.values():
            if isinstance(v, list):
                candidates = v
                break
        data = candidates or [data]

    index = {}
    for e in data:
        if isinstance(e, dict) and "image" in e:
            index[e["image"]] = e
    return index

def extract_ground_truth_homography(filename, metadata_or_path):
    base = os.path.basename(filename)
    index = _ensure_metadata_index(metadata_or_path)

    entry = index.get(base)
    if entry is None:
        print(f"[Warning] Ground truth not found for: {base}")
        return np.eye(3, dtype=float)

    H = entry.get("homography_matrix")
    if H is None:
        return np.eye(3, dtype=float)

    return np.array(H, dtype=float)



def run_homography_benchmark(image_folder, method, reference_path, metadata):
    results = []
    reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        raise FileNotFoundError(f"Reference image not found at: {reference_path}")

    img_labels, e_list = [], []

    for image_name in os.listdir(image_folder):
        if method == "qrcode" and "qrcode" not in image_name.lower():
            continue
        if method == "Ghost_seal" and "ghost_seal" not in image_name.lower():
            continue
        if "homography" not in image_name or not image_name.endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_folder, image_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # récupérer la ground truth et les angles depuis metadata
        entry = _ensure_metadata_index(metadata).get(os.path.basename(image_name))
        if entry is None:
            print(f"[Warning] No metadata for {image_name}")
            ground_truth_H = np.eye(3)
            label = image_name
        else:
            ground_truth_H = np.array(entry.get("homography_matrix", np.eye(3)), dtype=float)
            angle_x = entry.get("angle_x")
            angle_y = entry.get("angle_y")
            label = f"{angle_x}_{angle_y}"

        if method == "Ghost_seal":
            try:
                print("method Ghost_seal in process")
                final_image, total_h_estim = rectif.rectification(
                    reference_image, image, (0, -30), (30, 0),
                    4, "gaussian", 130, 130, 30, sliding=False, step=30
                )
                estimated_H = np.linalg.inv(total_h_estim)
            except Exception as e:
                print(f"Error estimating ghostseal H for {image_name}: {e}")
                estimated_H = None
        elif method == "qrcode":
            estimated_H = estimate_qrcode_homography(image)
        else:
            continue

        err = None
        if estimated_H is not None and estimated_H.shape == (3, 3):
            err = erreur_frobenius(estimated_H, ground_truth_H)
            img_labels.append(label)
            e_list.append(err)


        results.append({
            "image": image_name,
            "label": label,
            "method": method,
            "error": e_list,
            "success": estimated_H is not None and err is not None
        })

    if img_labels:
        rows = []
        index_meta = _ensure_metadata_index(metadata)
        for r in results:
            if not r["success"]:
                continue
            base = os.path.basename(r["image"])
            meta = index_meta.get(base, {})
            ax = meta.get("angle_x", None)
            ay = meta.get("angle_y", None)
            if ax is None or ay is None:
                continue
            err = r["error"]
            rows.append({
                "angle_x": float(ax),
                "angle_y": float(ay),
                "err": float(err),
                
            })

        if not rows:
            print("[Info] Aucun point à tracer (angles manquants ou estimations échouées).")
            return results

        df = pd.DataFrame(rows)

        fig, axes = plt.subplots(1, 1, figsize=(11, 11))


        sc1 = axes[0].scatter(
            df["angle_x"], df["angle_y"],
            c=df["err"], s=40, vmin=0.0, vmax=0.1
        )
        axes[0].set_title("Erreur frobenius")
        axes[0].set_xlabel("angle_x (°)")
        axes[0].set_ylabel("angle_y (°)")
        cbar1 = fig.colorbar(sc1, ax=axes[0])
        cbar1.set_label("err")


        fig.suptitle(f"Erreurs par angle — méthode: {method}")
        plt.tight_layout()

        os.makedirs("out", exist_ok=True)
        out_path = os.path.join("out", "errors_plot.png")
        fig.savefig(out_path, dpi=300)
        print("Graph enregistré en PNG :", os.path.abspath(out_path))

        plt.show()
        plt.close(fig)

    return results






image_folder = "/Users/i.bencheikh/Desktop/ENS/Doctorat/GhostSeal/GS3D/images/blur/qrcode/qrcode"
method = "qrcode"
reference_path = "/Users/i.bencheikh/Desktop/ENS/Doctorat/GhostSeal/GS3D/images/blur/qrcode/qrcode/qrcode_clean.png"
metadata = "/Users/i.bencheikh/Desktop/ENS/Doctorat/GhostSeal/GS3D/images/blur/qrcode/qrcode/metadata.json"
#run_homography_benchmark(image_folder, method, reference_path, metadata)
results = run_homography_benchmark(image_folder, method, reference_path, metadata)

# Sauvegarde en JSON
import json, os
os.makedirs("out_qrcode_6", exist_ok=True)
with open("out_qrcode_6/results_qrcode_6.json", "w") as f:
    json.dump(results, f, indent=4)  






