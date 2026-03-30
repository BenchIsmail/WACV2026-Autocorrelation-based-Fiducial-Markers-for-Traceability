import os
import cv2
import numpy as np
import json
from distortions import add_noise, add_blur, apply_occlusion,apply_jpeg_compression
import operations as oprt
from ghostseal_generator import generate_white_noise_and_shifts, gen_test
from qr_code import generate_qrcode,qr_chessboard
from deformation_generator import general_deformer



def generate_dataset(
    shape=512,
    output_dir="/images",
    marker_types=("qrcode", "Ghost_seal_noisy", "Ghost_seal_binary"),
    distortions=("none", "noise", "blur", "homography", "occlusion", "compression"),
    blur_levels=range(0),
    noise_levels=range(0),
    x_view_levels=range(0),
    y_view_levels=range(0),
    occlusion_σ=range(0),
    occlusion_q=range(0),
    compression_quality=range(0),
):
    print("CWD =", os.getcwd())
    print("output_dir (absolu) =", os.path.abspath(output_dir))

    os.makedirs(output_dir, exist_ok=True)
    results = []

    base_images = {}
    for marker_type in marker_types:
        if marker_type == "qrcode":
            img = generate_qrcode(marker_size=shape)

        elif marker_type == "chessboard_qrcode":
            img = qr_chessboard(rows = 3,
                  cols = 3,
                  cell_size = 256,
                  margin = 40,
                  per_cell_payload = True,
                  payload_prefix = "cell")
            
        elif marker_type == "Ghost_seal_noisy":
            height, width = shape, shape
            large_noise = generate_white_noise_and_shifts(height, width, (22, 0), (0, 22), seed=12532146)
            img = large_noise

        elif marker_type == "Ghost_seal_binary":
            height, width, dilation_size_i, density, angle_shift, norm_shift = shape,shape,1,0.005,90,22
            large_noise = gen_test(height, width, dilation_size_i, density, angle_shift, norm_shift)
            img = large_noise

        else:
            continue

        base_images[marker_type] = img

        clean_path = os.path.join(output_dir, f"{marker_type}_clean.png")
        print(clean_path)
        cv2.imwrite(clean_path, oprt.gray_to_png(img))

        results.append({
            "image": f"{marker_type}_clean.png",
            "marker_type": marker_type,
            "distortion": "none",
            "level": 0,
            "angle_x": None,
            "angle_y": None,
            "occlusion_fraction": None,
            "homography_matrix": None
        })

    for marker_type in marker_types:
        base_img = base_images[marker_type]

        for distortion in distortions:
            if distortion == "none":
                continue

            if distortion == "blur":
                levels = blur_levels
            elif distortion == "noise":
                levels = noise_levels
            elif distortion == "homography":
                levels = [(x, y) for x in x_view_levels for y in y_view_levels]
            elif distortion == "occlusion":
                levels = [(x, y) for x in occlusion_σ for y in occlusion_q]
            elif distortion == "compression":
                levels = list(compression_quality)  
            else:
                levels = [0]



            N_VARIANTS = 1
            for level in levels:
                for i in range(N_VARIANTS):
                    img = base_img.copy()
                    matrix_serialized = None
                    angle_x = angle_y = None
                    occlusion_used = None

                    if distortion == "noise":
                        img = add_noise(img, level)
                    elif distortion == "blur":
                        img = add_blur(img, level)
                    elif distortion == "homography":
                        angle_x, angle_y = level
                        warped, matrix = general_deformer(img,1, 0, 1, 0, angle_x, angle_y)
                        img = warped
                        matrix_serialized = matrix.tolist()
                        level_str = f"{angle_x}_{angle_y}"

                    if distortion == "occlusion":
                        σ,q = level
                        img = apply_occlusion(img, σ, q)
                        level_str = str(level).replace(".", "p")

                    else:
                        level_str = str(level)

                    filename = f"{marker_type}_{distortion}{level_str}_{i}.png"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, oprt.gray_to_png(img))

                    if distortion == "compression":
                        quality = int(level)  # niveau scalaire
                        img_1 = apply_jpeg_compression(img, quality)
                        img = img - img_1
                        level_str = f"q{int(level)}" 
                        filename = f"{marker_type}_{distortion}{level_str}_{i}.jpeg"
                        filepath = os.path.join(output_dir, filename)
                        cv2.imwrite(filepath, oprt.gray_to_png(img))

                    results.append({
                        "image": filename,
                        "marker_type": marker_type,
                        "distortion": distortion,
                        "level": level if not isinstance(level, tuple) else None,
                        "angle_x": angle_x,
                        "angle_y": angle_y,
                        "occlusion_fraction": occlusion_used,
                        "homography_matrix": matrix_serialized,
                        "jpeg_quality": int(level) if distortion == "compression" else None,
                    })


    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    from pathlib import Path
    SCRIPT_DIR = Path(__file__).resolve().parent

    generate_dataset(
        shape=1024,
        output_dir = SCRIPT_DIR / "images/occlusion/chessboard_qrcode",#"qrcode", "chessboard_qrcode", "Ghost_seal_noisy", "Ghost_seal_binary"
        marker_types=("chessboard_qrcode", ),#"qrcode", "chessboard_qrcode", "Ghost_seal_noisy", "Ghost_seal_binary"
        distortions=("occlusion",),#"none", "noise", "blur", "homography", "occlusion","compression"
        blur_levels=range(0, 100),
        noise_levels=range(0, 100),
        x_view_levels=range(-40, 40,2),
        y_view_levels=range(-40, 40,2),
        occlusion_σ=range(0,101,5),
        occlusion_q=range(0,101,5),
        compression_quality=range(0,101),
    )
