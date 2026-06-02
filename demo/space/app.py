import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModel


def find_project_root() -> Path:
    """Find a directory containing the local Whisper-Pinyin source tree."""
    app_dir = Path(__file__).resolve().parent
    for candidate in (app_dir, *app_dir.parents):
        if (candidate / "whisper").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find the Whisper-Pinyin source tree. Upload the `whisper/` "
        "directory with this Space app."
    )


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))


MODEL_REPO_ID = os.getenv("WHISPER_PINYIN_MODEL_ID", "walston/whisper-pinyin")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    model = AutoModel.from_pretrained(
        MODEL_REPO_ID,
        trust_remote_code=True,
    )
    model.to(DEVICE)
    model.eval()
    return model


def decode_audio(audio_path: str):
    if not audio_path:
        raise gr.Error("Please upload an audio file first.")

    model = load_model()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    result = model.decode_file(audio_path, device=DEVICE)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    decode_time = time.perf_counter() - start_time
    rtf = decode_time / max(result["duration"], 1e-6)
    info = (
        f"Device: {DEVICE}\n"
        f"Model repo: {MODEL_REPO_ID}\n"
        f"Input duration: {result['duration']:.2f}s\n"
        f"Decode time: {decode_time:.2f}s\n"
        f"RTF: {rtf:.2f}"
    )
    return result["pinyin"], result["tokens"], info


with gr.Blocks(title="Whisper-Pinyin Demo") as demo:
    gr.Markdown(
        "# Whisper-Pinyin\n"
        "Upload Mandarin speech audio to decode Pinyin with the cross-augmentation "
        "continuous checkpoint."
    )
    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio",
        )
        with gr.Column():
            syllable_output = gr.Textbox(label="Decoded Pinyin", lines=4)
            token_output = gr.Textbox(label="Raw token sequence", lines=4)
            info_output = gr.Textbox(label="Runtime info", lines=5)

    decode_button = gr.Button("Decode", variant="primary")
    decode_button.click(
        fn=decode_audio,
        inputs=audio_input,
        outputs=[syllable_output, token_output, info_output],
    )


if __name__ == "__main__":
    demo.queue().launch()
