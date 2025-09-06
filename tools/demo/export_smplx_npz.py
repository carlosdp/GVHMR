#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    # Fallback for lists or other array-like structures
    return np.asarray(x)


def normalize_betas(betas: np.ndarray, num_frames: int) -> np.ndarray:
    """Ensure betas shape is (F, 10). If (10,), tile across frames."""
    if betas.ndim == 1:
        betas = np.tile(betas[None, :], (num_frames, 1))
    return betas


def zero_array(num_frames: int, dim: int) -> np.ndarray:
    return np.zeros((num_frames, dim), dtype=np.float32)


def extract_smplx_params(pred: Dict[str, Any], space: str) -> Dict[str, np.ndarray]:
    assert space in {"incam", "global"}
    key = f"smpl_params_{space}"
    assert key in pred, (
        f"Key '{key}' not found in input file. Available keys: {list(pred.keys())}"
    )

    params = pred[key]
    # Expected keys inside params: body_pose(F,63), betas(F,10 or 10), global_orient(F,3), transl(F,3)
    required = ["body_pose", "betas", "global_orient"]
    for rk in required:
        assert rk in params, f"Parameter '{rk}' missing in '{key}'"

    body_pose = to_numpy(params["body_pose"]).astype(np.float32)
    global_orient = to_numpy(params["global_orient"]).astype(np.float32)
    transl = to_numpy(params.get("transl", None))
    if transl is None:
        transl = np.zeros((body_pose.shape[0], 3), dtype=np.float32)
    else:
        transl = transl.astype(np.float32)

    betas = to_numpy(params["betas"]).astype(np.float32)
    num_frames = body_pose.shape[0]
    betas = normalize_betas(betas, num_frames)

    return {
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": transl,
        "betas": betas,
    }


def maybe_camera_intrinsics(pred: Dict[str, Any]) -> np.ndarray:
    if "K_fullimg" not in pred:
        return None
    K = to_numpy(pred["K_fullimg"]).astype(np.float32)
    # Expect shape (F, 3, 3)
    return K


def build_smplx_npz_dict(
    base_params: Dict[str, np.ndarray], include_hands_face: bool = True
) -> Dict[str, np.ndarray]:
    """Construct a SMPLX-style .npz dictionary.

    Keys follow common SMPLX parameter naming used by many toolchains:
      - global_orient: (F, 3)
      - body_pose: (F, 63)
      - betas: (F, 10)
      - transl: (F, 3)
      - left_hand_pose: (F, 45)
      - right_hand_pose: (F, 45)
      - jaw_pose: (F, 3)
      - leye_pose: (F, 3)
      - reye_pose: (F, 3)
      - expression: (F, 10)
    """
    F = base_params["body_pose"].shape[0]

    out = {
        "global_orient": base_params["global_orient"],
        "body_pose": base_params["body_pose"],
        "betas": base_params["betas"],
        "transl": base_params["transl"],
    }

    if include_hands_face:
        out.update(
            {
                "left_hand_pose": zero_array(F, 45),
                "right_hand_pose": zero_array(F, 45),
                "jaw_pose": zero_array(F, 3),
                "leye_pose": zero_array(F, 3),
                "reye_pose": zero_array(F, 3),
                "expression": zero_array(F, 10),
            }
        )

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert GVHMR demo .pt outputs to SMPLX .npz"
    )
    parser.add_argument(
        "--input_pt",
        type=str,
        required=True,
        help="Path to outputs/demo/.../hmr4d_results.pt",
    )
    parser.add_argument(
        "--output_npz", type=str, required=True, help="Path to save SMPLX params .npz"
    )
    parser.add_argument(
        "--space",
        type=str,
        default="incam",
        choices=["incam", "global"],
        help="Choose which coordinate system to export: 'incam' (camera) or 'global'",
    )
    parser.add_argument(
        "--include_camera",
        action="store_true",
        help="If set, also include K_fullimg (F,3,3) in the .npz under key 'K_fullimg'",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default=None,
        choices=["neutral", "male", "female"],
        help="Optional gender tag to include in the .npz (string scalar)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS metadata to include in the .npz (scalar)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_pt)
    assert input_path.exists(), f"Input .pt file not found: {input_path}"

    pred = torch.load(str(input_path))
    base_params = extract_smplx_params(pred, args.space)

    out_dict = build_smplx_npz_dict(base_params, include_hands_face=True)

    if args.include_camera:
        K = maybe_camera_intrinsics(pred)
        if K is not None:
            out_dict["K_fullimg"] = K

    if args.gender is not None:
        out_dict["gender"] = np.array(args.gender)
    if args.fps is not None:
        out_dict["fps"] = np.array(args.fps, dtype=np.float32)

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(str(output_path), **out_dict)
    print(f"Saved SMPLX .npz to: {output_path}")


if __name__ == "__main__":
    main()
