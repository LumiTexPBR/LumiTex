import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import clip
import lpips
import numpy as np
import torch
import torchvision.transforms as transforms
from cleanfid import fid
from PIL import Image


UV_DATA_PATH = "testset"
RESULT_DETAIL_PATH = "texture_scores_dict.json"
RESULT_AVG_PATH = "texture_final_avg_scores.json"
CASE_START = 1
CASE_END = 133
# Methods to evaluate.
BASELINE_METHODS = [
    "ours_250707",
    "dreammat",
    "MV-Adapter",
    "paint_it",
    "Step1X-3D",
    "Hunyuan3D-2.1",
    "syncmvd",
    "unitex",
]
# Supported benchmark metrics.
BENCHMARK_METRICS = ["FID", "CLIP-FID", "CMMD", "CLIP-I", "LPIPS"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LPIPS_MODEL = lpips.LPIPS(net="alex").to(DEVICE).eval()
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
CLIP_MODEL.eval()
LPIPS_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=Image.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [0, 1] -> [-1, 1]
    ]
)


def initialize_scores_dict(
    methods: List[str], metrics: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Initialize nested score storage as {method: {metric: {case_name: score}}}.
    """
    return {method: {metric: {} for metric in metrics} for method in methods}


@contextmanager
def temporary_metric_dirs(
    img1_path: Path, img2_path: Path
) -> Iterator[Tuple[str, str]]:
    """
    Create and clean temporary folders that each contain one texture image.
    """
    temp_dir1 = Path(tempfile.mkdtemp())
    temp_dir2 = Path(tempfile.mkdtemp())
    try:
        shutil.copy(str(img1_path), str(temp_dir1 / "material_0.png"))
        shutil.copy(str(img2_path), str(temp_dir2 / "material_0.png"))
        yield str(temp_dir1), str(temp_dir2)
    finally:
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)


def calculate_fid_single_texture(model_texture_path: str, gt_texture_path: str) -> float:
    """
    Compute FID for one predicted texture against one ground-truth texture.
    """
    with temporary_metric_dirs(Path(model_texture_path), Path(gt_texture_path)) as (
        temp_dir1,
        temp_dir2,
    ):
        return float(fid.compute_fid(temp_dir1, temp_dir2))


def calculate_clip_fid_single_texture(model_texture_path: str, gt_texture_path: str) -> float:
    """
    Compute CLIP-FID for one predicted texture against one ground-truth texture.
    """
    with temporary_metric_dirs(Path(model_texture_path), Path(gt_texture_path)) as (
        temp_dir1,
        temp_dir2,
    ):
        return float(fid.compute_fid(temp_dir1, temp_dir2, mode="clean", model_name="clip_vit_b_32"))


def calculate_cmmd_single_texture(model_texture_path: str, gt_texture_path: str) -> float:
    """
    Compute CMMD for one predicted texture against one ground-truth texture.
    """
    from cmmd.main import compute_cmmd

    with temporary_metric_dirs(Path(model_texture_path), Path(gt_texture_path)) as (
        temp_dir1,
        temp_dir2,
    ):
        return float(compute_cmmd(temp_dir1, temp_dir2, None, 32, -1))


def calculate_clip_i_single_texture(model_texture_path: str, gt_texture_path: str) -> float:
    """
    Compute CLIP image-to-image cosine similarity for one texture pair.
    """
    img1 = Image.open(Path(model_texture_path)).convert("RGB")
    img2 = Image.open(Path(gt_texture_path)).convert("RGB")

    img1_tensor = CLIP_PREPROCESS(img1).unsqueeze(0).to(DEVICE)
    img2_tensor = CLIP_PREPROCESS(img2).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img1_feat = CLIP_MODEL.encode_image(img1_tensor)
        img2_feat = CLIP_MODEL.encode_image(img2_tensor)
        clip_score = torch.nn.functional.cosine_similarity(img1_feat, img2_feat, dim=1)

    return float(clip_score.detach().cpu().numpy()[0])


def calculate_lpips_single_texture(model_texture_path: str, gt_texture_path: str) -> float:
    """
    Compute LPIPS distance for one predicted texture against one ground-truth texture.
    """
    img1 = Image.open(Path(model_texture_path)).convert("RGB")
    img2 = Image.open(Path(gt_texture_path)).convert("RGB")
    img1_tensor = LPIPS_PREPROCESS(img1).unsqueeze(0).to(DEVICE)
    img2_tensor = LPIPS_PREPROCESS(img2).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        lpips_score = LPIPS_MODEL(img1_tensor, img2_tensor)

    return float(lpips_score.item())


METRIC_FUNCTIONS: Dict[str, Callable[[str, str], float]] = {
    "FID": calculate_fid_single_texture,
    "CLIP-FID": calculate_clip_fid_single_texture,
    "CMMD": calculate_cmmd_single_texture,
    "CLIP-I": calculate_clip_i_single_texture,
    "LPIPS": calculate_lpips_single_texture,
}


def compute_metric(metric: str, method_texture_path: str, gt_texture_path: str) -> float:
    """
    Dispatch metric computation to the corresponding single-texture metric function.
    """
    metric_fn = METRIC_FUNCTIONS.get(metric)
    if metric_fn is None:
        raise ValueError(f"Unsupported metric: {metric}")
    return metric_fn(method_texture_path, gt_texture_path)


def convert_numpy(value: Any) -> Any:
    """
    Convert NumPy scalar/array values to JSON-serializable Python values.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def evaluate_all_cases(
    uv_data_path: str,
    methods: List[str],
    metrics: List[str],
    case_start: int,
    case_end: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run benchmark metrics for all case/method pairs and return raw per-case scores.
    """
    scores_dict = initialize_scores_dict(methods, metrics)

    print("Starting texture evaluation...")
    dataset_root = Path(uv_data_path)
    for idx in range(case_start, case_end + 1):
        case_name = f"case{idx}"
        case_path = dataset_root / case_name

        if not case_path.exists():
            print(f"Skipping {case_name} - directory does not exist")
            continue

        gt_texture_path = case_path / "mesh" / "material_0.png"
        if not gt_texture_path.exists():
            print(f"Skipping {case_name} - GT texture does not exist")
            continue

        print(f"\nProcessing {case_name}...")
        for method in methods:
            method_texture_path = case_path / method / "material_0.png"
            if not method_texture_path.exists():
                print(f"  Skipping {method} - texture does not exist")
                continue

            print(f"  Evaluating {method}...")
            for metric in metrics:
                try:
                    score = compute_metric(metric, str(method_texture_path), str(gt_texture_path))
                    scores_dict[method][metric][case_name] = score
                    print(f"    {method} {metric}: {score:.4f}")
                except Exception as error:
                    print(
                        f"    Error calculating {metric} for {method} on {case_name}: {error}"
                    )

    return scores_dict


def compute_average_scores(
    methods: List[str],
    metrics: List[str],
    scores_dict: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-method average scores across available cases for each metric.
    """
    print("\n=== Computing Average Scores ===")
    final_avg_scores: Dict[str, Dict[str, float]] = {}

    for method in methods:
        final_avg_scores[method] = {}
        print(f"\n{method}:")
        for metric in metrics:
            scores = list(scores_dict[method][metric].values())
            if scores:
                avg_score = float(np.mean(np.array(scores, dtype=np.float32)))
                final_avg_scores[method][metric] = avg_score
                print(f"  {metric}: {avg_score:.4f} (from {len(scores)} cases)")
            else:
                final_avg_scores[method][metric] = None
                print(f"  {metric}: No valid scores")

    return final_avg_scores


def save_results(
    scores_dict: Dict[str, Dict[str, Dict[str, float]]],
    final_avg_scores: Dict[str, Dict[str, float]],
    detail_path: str,
    avg_path: str,
) -> None:
    """
    Save detailed and averaged benchmark results as JSON files.
    """
    print("\n=== Saving Results ===")

    with open(detail_path, "w", encoding="utf-8") as file:
        json.dump(scores_dict, file, default=convert_numpy, indent=4)
    print(f"Detailed scores saved to: {detail_path}")

    with open(avg_path, "w", encoding="utf-8") as file:
        json.dump(final_avg_scores, file, default=convert_numpy, indent=4)
    print(f"Average scores saved to: {avg_path}")


def main() -> None:
    """
    Execute the full texture benchmark pipeline from evaluation to report export.
    """
    scores_dict = evaluate_all_cases(
        uv_data_path=UV_DATA_PATH,
        methods=BASELINE_METHODS,
        metrics=BENCHMARK_METRICS,
        case_start=CASE_START,
        case_end=CASE_END,
    )
    final_avg_scores = compute_average_scores(BASELINE_METHODS, BENCHMARK_METRICS, scores_dict)
    save_results(scores_dict, final_avg_scores, RESULT_DETAIL_PATH, RESULT_AVG_PATH)


if __name__ == "__main__":
    main()
