"""Streamlit demo for frozen low-light enhancement methods."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from lowlight.demo.inference import (  # noqa: E402
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE,
    MAX_UPLOAD_BYTES,
    METHOD_SPECS,
    SUPPORTED_SEEDS,
    DemoError,
    InferenceResult,
    NiqeUnavailableError,
    UploadedImage,
    image_to_png_bytes,
    load_engine,
    load_niqe,
    load_uploaded_image,
    method_spec,
    run_engine,
    score_niqe,
)

st.set_page_config(
    page_title="Low-light enhancement demo",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _cached_engine(method_key: str, seed: int, device: str):
    return load_engine(method_key, seed=seed, device=device)


@st.cache_resource(show_spinner=False)
def _cached_niqe(device: str):
    return load_niqe(device)


def _method_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for spec in METHOD_SPECS.values():
        groups.setdefault(spec.group, []).append(spec.key)
    return groups


def _display_method(method_key: str, seed: int) -> None:
    spec = method_spec(method_key)
    st.caption(f"{spec.stage} · {spec.description}")
    if spec.config:
        st.code(spec.config, language="text")
    elif spec.checkpoint_root:
        st.code(f"{spec.checkpoint_root}/seed{seed}/best.pt", language="text")
    if spec.seedable:
        st.caption(f"Checkpoint seed: {seed}")
    else:
        st.caption("This method has no seeded checkpoint.")


def _render_niqe(
    source: UploadedImage,
    result: InferenceResult,
    device: str,
    cache_key: tuple[object, ...],
) -> None:
    st.subheader("No-reference NIQE")
    cached = st.session_state.get("demo_niqe")
    if isinstance(cached, dict) and cached.get("key") == cache_key:
        original = float(cached["original"])
        enhanced = float(cached["enhanced"])
    else:
        try:
            metric = _cached_niqe(device)
            original = score_niqe(source.image, metric=metric)
            enhanced = score_niqe(result.image, metric=metric)
        except NiqeUnavailableError as error:
            st.warning(f"NIQE unavailable: {error}")
            return
        except Exception as error:
            st.warning(f"NIQE unavailable: {error}")
            return
        st.session_state["demo_niqe"] = {
            "key": cache_key,
            "original": original,
            "enhanced": enhanced,
        }
    left, right = st.columns(2)
    left.metric("Original NIQE", f"{original:.3f}")
    right.metric("Enhanced NIQE", f"{enhanced:.3f}")
    st.caption(
        "NIQE is lower-is-better and no-reference. It is not a substitute for "
        "paired PSNR/SSIM/LPIPS on the locked test sets."
    )


st.title("Low-light enhancement demo")
st.warning(
    "This is an interactive demonstration, not a benchmark. One uploaded image "
    "has no ground truth; the reported benchmark results come from the locked "
    "test sets and are not reproduced here."
)

with st.sidebar:
    st.header("Method")
    groups = _method_groups()
    group = st.selectbox("Family", list(groups), index=0)
    labels = {key: METHOD_SPECS[key].label for key in groups[group]}
    method_key = st.selectbox(
        "Configuration",
        groups[group],
        format_func=lambda key: labels[key],
    )
    spec = method_spec(method_key)
    seed = (
        st.selectbox("Checkpoint seed", SUPPORTED_SEEDS, index=0)
        if spec.seedable
        else 0
    )
    device = st.selectbox(
        "Device",
        ["auto", "cpu"],
        index=0,
        help="auto uses CUDA when torch reports it is available.",
    )
    st.divider()
    st.subheader("Input limits")
    st.write(f"PNG or JPEG, at most {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    st.write(f"Maximum side: {MAX_IMAGE_SIDE} px.")
    st.write(f"Maximum pixels: {MAX_IMAGE_PIXELS:,}.")
    st.caption(
        "Images are converted to RGB and kept in [0, 1]. Grayscale uploads are "
        "promoted to RGB. No resizing or cropping is applied."
    )
    st.divider()
    st.subheader("Checkpoints")
    st.caption(
        "Weights are not stored in git. If a checkpoint is missing, train it from "
        "the commands in README.md or select another method."
    )

_display_method(method_key, seed)

uploaded = st.file_uploader(
    "Upload a low-light image",
    type=["png", "jpg", "jpeg"],
    max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
)
if uploaded is None:
    st.info(
        "Upload a PNG or JPEG. The demo validates the format and dimensions, "
        "then runs the selected frozen configuration."
    )
    with st.expander("Demo limitations"):
        st.markdown(
            "- The demo is not a benchmark and has no reference image.\n"
            "- NIQE is no-reference and may disagree with perceptual quality.\n"
            "- A model trained for denoising is not a complete brightener, and "
            "vice versa.\n"
            "- Results on out-of-domain images can be worse than the reported "
            "test results.\n"
            "- Raw data, outputs and checkpoints are intentionally not in git."
        )
    st.stop()

try:
    source = load_uploaded_image(uploaded.getvalue(), filename=uploaded.name)
except DemoError as error:
    st.error(str(error))
    st.stop()

data_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
signature = (data_hash, method_key, seed, device)

left, right = st.columns(2)
with left:
    st.subheader("Original")
    st.image(source.image, width="stretch")
    st.caption(
        f"{source.width}x{source.height} · {source.format} · "
        f"source mode {source.source_mode} · RGB after preprocessing"
    )
with right:
    st.subheader("Enhanced")
    placeholder = st.empty()
    placeholder.info("Run the selected method to see the output.")

run_clicked = st.button("Enhance image", type="primary")

if run_clicked:
    try:
        with st.spinner("Loading frozen checkpoint and running inference..."):
            engine = _cached_engine(method_key, seed, device)
            result = run_engine(engine, source.image)
        st.session_state["demo_signature"] = signature
        st.session_state["demo_result"] = result
        st.session_state.pop("demo_niqe", None)
    except DemoError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error(f"Inference failed: {error}")
        with st.expander("Error details"):
            st.exception(error)
        st.stop()

if st.session_state.get("demo_signature") != signature:
    st.stop()

stored_result = st.session_state.get("demo_result")
if not isinstance(stored_result, InferenceResult):
    st.stop()
result = stored_result

with right:
    placeholder.empty()
    st.image(result.image, width="stretch")
    st.caption(
        f"{result.method.label} · seed {result.seed} · {result.device}"
    )

st.subheader("Timing and output")
metrics = st.columns(4)
metrics[0].metric("Total inference", f"{result.total_ms:,.0f} ms")
metrics[1].metric("Output size", f"{result.image.shape[1]}x{result.image.shape[0]}")
metrics[2].metric("Method", result.method.group)
metrics[3].metric("Seed", str(result.seed) if result.method.seedable else "n/a")
if result.component_ms:
    st.caption(
        "Components: "
        + " · ".join(
            f"{name} {milliseconds:,.0f} ms"
            for name, milliseconds in result.component_ms.items()
        )
    )
st.caption(
    "Timing is one wall-clock run on this machine and is not a controlled "
    "benchmark. The first run can include checkpoint loading."
)

st.download_button(
    "Download enhanced PNG",
    data=image_to_png_bytes(result.image),
    file_name=f"{Path(uploaded.name).stem}_enhanced.png",
    mime="image/png",
)

_render_niqe(source, result, result.device, signature)

with st.expander("Method details"):
    st.json(result.pipeline_metadata)

st.divider()
st.subheader("Known limitations")
st.markdown(
    "- The demo does not train, download data or modify experiment configs.\n"
    "- Uploaded images are not from the locked test splits, so no PSNR, SSIM or "
    "LPIPS can be computed.\n"
    "- BM3D is comparatively slow on CPU; U-Net checkpoints require local weights.\n"
    "- Failures and negative results from the full study remain in the Stage 07 "
    "handout."
)
