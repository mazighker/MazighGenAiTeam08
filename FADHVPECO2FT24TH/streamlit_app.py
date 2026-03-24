import os
import math
import uuid
import tempfile
from pathlib import Path
from datetime import datetime
import gzip
import io
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import pathlib
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import dct
import pathlib
import altair as alt
import base64
import tempfile

from snowflake.snowpark.context import get_active_session

# =========================================================
# CONFIG
# =========================================================
TARGET_SR = 22050
SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_SECONDS

N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
N_MELS = 128
N_MFCC = 20
FMIN = 20.0
FMAX = TARGET_SR / 2.0

BASELINE_MODEL_STAGE_FILE = "@model_stage/respiratory_prediction_model_mfcc2d.pth.gz"
MULTIMODAL_MODEL_STAGE_FILE = "@model_stage/respirasense_multimodal_final.pth.gz"

DEFAULT_LABELS = ["Asthma", "Bronchial", "Copd", "Healthy", "Pneumonia"]

PREDICTIONS_TABLE = "RESP_APP_PREDICTIONS"
SAVE_PREDICTIONS = True

def load_css(path: str):
    css = pathlib.Path(path).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("style.css")
def load_image_from_stage(stage_path: str) -> str:
    """Charge une image depuis un stage Snowflake et retourne un base64 data URI."""
    session = get_active_session()
    with tempfile.TemporaryDirectory() as tmpdir:
        session.file.get(stage_path, tmpdir)
        filename = Path(stage_path).name
        local_path = list(Path(tmpdir).rglob(filename))[0]
        with open(local_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(stage_path).suffix.lstrip(".").lower()
    mime = "image/png" if ext == "png" else f"image/{ext}"
    return f"data:{mime};base64,{data}"
logo_center = load_image_from_stage("@LOGOS_stage/1000033098.png")
st.set_page_config(page_title="RespiraSense AI", page_icon=logo_center, layout="wide")




# ── Chargement des logos ──
logo_left  = load_image_from_stage("@LOGOS_stage/Université_Paris_8_Logo_2024.svg.png")
logo_right = load_image_from_stage("@LOGOS_stage/LOGO-TESSAN-COMPLET-FR-1024x346.png")

# ── Header avec les 2 logos encadrant le titre ──
st.markdown(f"""
<div style="
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 8px 0 4px 0;
">
  <img src="{logo_left}" style="height: 80px; object-fit: contain; flex-shrink: 0;" />
  <div style="flex: 1; text-align: center;">
    <h1 style="margin: 0; padding: 0; font-size: clamp(1.6rem, 3.5vw, 3.2rem); line-height: 1.2;">
      <span style="display: inline-block; position: relative;">
        🫁
        <div style="
          position: absolute;
          bottom: -10px;
          left: 50%;
          transform: translateX(-50%);
          width: 36px;
          height: 3px;
          background: linear-gradient(90deg, #1a9460, #34c98a);
          border-radius: 99px;
        "></div>
      </span>
      RespiraSense AI
    </h1>
  </div>
  <img src="{logo_right}" style="height: 70px; object-fit: contain; flex-shrink: 0;" />
</div>
""", unsafe_allow_html=True)
st.markdown("---")
#st.title("🫁 Respiratory AI Assistant")
#st.caption("Analyse respiratoire dans Snowflake : mode audio seul ou multimodal")

# =========================================================
# MODELS
# =========================================================
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights


class MelCNNBiLSTMAttention(nn.Module):
    def __init__(self, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm_hidden = lstm_hidden
        self.lstm = None
        self.attention = None
        self.output_dim = 2 * lstm_hidden

    def _build_sequence_modules_if_needed(self, x):
        if self.lstm is None:
            _, c, h, w = x.shape
            self.lstm = nn.LSTM(
                input_size=c * h,
                hidden_size=self.lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.attention = AttentionLayer(2 * self.lstm_hidden)
            self.lstm.to(x.device)
            self.attention.to(x.device)

    def forward(self, mel):
        x = self.cnn(mel)
        self._build_sequence_modules_if_needed(x)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        context, attn_weights = self.attention(x)
        return context, attn_weights


class MFCCCNNBranch(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = 128

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return x


class HybridAudioModel(nn.Module):
    def __init__(self, num_classes, lstm_hidden=128, fusion_hidden=128, dropout=0.3):
        super().__init__()
        self.mel_branch = MelCNNBiLSTMAttention(lstm_hidden=lstm_hidden, dropout=dropout)
        self.mfcc_branch = MFCCCNNBranch(dropout=dropout)

        fusion_input_dim = self.mel_branch.output_dim + self.mfcc_branch.output_dim  # 256 + 128 = 384

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, mel, mfcc):
        mel_vec, attn_weights = self.mel_branch(mel)
        mfcc_vec = self.mfcc_branch(mfcc)
        fused = torch.cat([mel_vec, mfcc_vec], dim=1)
        logits = self.classifier(fused)
        return logits, attn_weights


class PhysioBranch(nn.Module):
    def __init__(self, input_dim=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.output_dim = 64

    def forward(self, x):
        return self.net(x)


class MultimodalRespiraSense(nn.Module):
    def __init__(self, num_classes, lstm_hidden=128, fusion_hidden=128, dropout=0.3, physio_dim=4):
        super().__init__()
        self.mel_branch = MelCNNBiLSTMAttention(lstm_hidden=lstm_hidden, dropout=dropout)
        self.mfcc_branch = MFCCCNNBranch(dropout=dropout)
        self.physio_branch = PhysioBranch(input_dim=physio_dim, dropout=0.2)

        fusion_input_dim = (
            self.mel_branch.output_dim +
            self.mfcc_branch.output_dim +
            self.physio_branch.output_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, mel, mfcc, physio):
        mel_vec, attn_weights = self.mel_branch(mel)
        mfcc_vec = self.mfcc_branch(mfcc)
        physio_vec = self.physio_branch(physio)
        fused = torch.cat([mel_vec, mfcc_vec, physio_vec], dim=1)
        logits = self.classifier(fused)
        return logits, attn_weights

# =========================================================
# AUDIO UTILS
# =========================================================
def to_float32_audio(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
        max_abs = np.max(np.abs(x)) if x.size else 0.0
        if max_abs > 1.5:
            x = x / max_abs
        return x
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if x.dtype == np.int32:
        return (x.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    if x.dtype == np.uint8:
        return ((x.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    x = x.astype(np.float32)
    max_abs = np.max(np.abs(x)) if x.size else 0.0
    return x / max_abs if max_abs > 0 else x


def convert_to_mono(x):
    return x if x.ndim == 1 else np.mean(x, axis=1).astype(np.float32)


def normalize_peak(x, eps=1e-8):
    peak = np.max(np.abs(x)) if x.size else 0.0
    return x.astype(np.float32) if peak < eps else (x / peak).astype(np.float32)


def resample_audio(x, orig_sr, target_sr):
    if orig_sr == target_sr:
        return x.astype(np.float32)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(x, up, down).astype(np.float32)


def pad_or_trim(x, target_len):
    n = len(x)
    if n == target_len:
        return x.astype(np.float32)
    if n > target_len:
        return x[:target_len].astype(np.float32)
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = x
    return out


def load_wav(local_path):
    sr, x = wavfile.read(local_path)
    x = to_float32_audio(x)
    x = convert_to_mono(x)
    return int(sr), x.astype(np.float32)


def is_valid_audio(x, sr):
    if x is None or len(x) == 0:
        return False
    if np.isnan(x).any():
        return False
    if np.max(np.abs(x)) < 1e-5:
        return False
    if sr < 4000 or sr > 96000:
        return False
    return True


def segment_audio(x, segment_len):
    if len(x) < segment_len:
        return [(0, 0, min(len(x), segment_len), pad_or_trim(x, segment_len))]
    segments = []
    start = 0
    seg_id = 0
    while start < len(x):
        end = start + segment_len
        chunk = x[start:end]
        if len(chunk) < segment_len:
            chunk = pad_or_trim(chunk, segment_len)
        segments.append((seg_id, start, min(end, len(x)), chunk.astype(np.float32)))
        seg_id += 1
        start += segment_len
    return segments


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filter_bank(sr, n_fft, n_mels, fmin, fmax):
    n_freqs = n_fft // 2 + 1
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        center = min(center + 1, n_freqs - 1) if center == left else center
        right = min(right + 1, n_freqs - 1) if right == center else right
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    enorm = 2.0 / np.maximum(hz_points[2:n_mels + 2] - hz_points[:n_mels], 1e-10)
    fb *= enorm[:, np.newaxis]
    return fb.astype(np.float32)


def stft_power(x, sr, n_fft, hop_length, win_length):
    freqs, times, zxx = signal.stft(
        x, fs=sr, window="hann", nperseg=win_length,
        noverlap=win_length - hop_length, nfft=n_fft,
        padded=True, boundary="zeros"
    )
    return freqs.astype(np.float32), times.astype(np.float32), (np.abs(zxx) ** 2).astype(np.float32)


def compute_mel_spectrogram(x):
    _, _, power = stft_power(x, TARGET_SR, N_FFT, HOP_LENGTH, WIN_LENGTH)
    fb = mel_filter_bank(TARGET_SR, N_FFT, N_MELS, FMIN, FMAX)
    mel_power = np.maximum(np.dot(fb, power), 1e-10)
    return (10.0 * np.log10(mel_power)).astype(np.float32)


def compute_mfcc_from_mel(mel_db):
    return dct(mel_db, type=2, axis=0, norm="ortho")[:N_MFCC].astype(np.float32)


def extract_features(segment):
    mel = compute_mel_spectrogram(segment)
    mfcc = compute_mfcc_from_mel(mel)
    return mel.astype(np.float32), mfcc.astype(np.float32)


def pad_or_trim_2d(x, target_h, target_w):
    h, w = x.shape
    out = np.zeros((target_h, target_w), dtype=np.float32)
    out[:min(h, target_h), :min(w, target_w)] = x[:min(h, target_h), :min(w, target_w)]
    return out

# =========================================================
# SNOWFLAKE
# =========================================================
@st.cache_resource(show_spinner=False)
def get_session():
    return get_active_session()


def ensure_predictions_table():
    session = get_session()
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
            PREDICTION_ID STRING,
            CREATED_AT TIMESTAMP_NTZ,
            FILE_NAME STRING,
            MODE_USED STRING,
            N_SEGMENTS INT,
            PREDICTED_LABEL STRING,
            CONFIDENCE FLOAT,
            PROBABILITIES_JSON STRING,
            SEGMENT_DETAILS_JSON STRING,
            RECOMMENDATION STRING
        )
    """).collect()

# =========================================================
# MODEL LOADERS
# =========================================================
def _load_checkpoint_from_stage(stage_file):
    session = get_session()
    with tempfile.TemporaryDirectory() as tmpdir:
        session.file.get(stage_file, tmpdir)
        model_name = Path(stage_file).name
        candidates = list(Path(tmpdir).rglob(model_name))
        if not candidates:
            raise FileNotFoundError(f"Impossible de charger le modèle depuis {stage_file}")
        local_model_path = str(candidates[0])

        if local_model_path.endswith(".gz"):
            with gzip.open(local_model_path, "rb") as gz:
                checkpoint = torch.load(io.BytesIO(gz.read()), map_location="cpu")
        else:
            checkpoint = torch.load(local_model_path, map_location="cpu")
    return checkpoint


@st.cache_resource(show_spinner=True)
def load_baseline_bundle():
    checkpoint = _load_checkpoint_from_stage(BASELINE_MODEL_STAGE_FILE)

    labels = checkpoint.get("label_classes", DEFAULT_LABELS)
    target_mel_shape = checkpoint.get("target_mel_shape", [128, 216])
    target_mfcc_shape = checkpoint.get("target_mfcc_shape", [20, 216])

    model = HybridAudioModel(
        num_classes=len(labels),
        lstm_hidden=128,
        fusion_hidden=128,
        dropout=0.3
    )

    dummy_mel = torch.zeros(1, 1, target_mel_shape[0], target_mel_shape[1], dtype=torch.float32)
    dummy_mfcc = torch.zeros(1, 1, target_mfcc_shape[0], target_mfcc_shape[1], dtype=torch.float32)
    with torch.no_grad():
        _ = model(dummy_mel, dummy_mfcc)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "labels": labels,
        "target_mel_shape": target_mel_shape,
        "target_mfcc_shape": target_mfcc_shape,
        "mel_mean": float(checkpoint.get("mel_mean", 0.0)),
        "mel_std": float(checkpoint.get("mel_std", 1.0)),
        "mfcc_full_mean": float(checkpoint.get("mfcc_full_mean", 0.0)),
        "mfcc_full_std": float(checkpoint.get("mfcc_full_std", 1.0)),
        "mode": "audio_only"
    }


@st.cache_resource(show_spinner=True)
def load_multimodal_bundle():
    checkpoint = _load_checkpoint_from_stage(MULTIMODAL_MODEL_STAGE_FILE)

    labels = checkpoint.get("label_classes", DEFAULT_LABELS)
    target_mel_shape = checkpoint.get("target_mel_shape", [128, 216])
    target_mfcc_shape = checkpoint.get("target_mfcc_shape", [20, 216])
    physio_dim = checkpoint.get("physio_dim", 4)

    model = MultimodalRespiraSense(
        num_classes=len(labels),
        lstm_hidden=128,
        fusion_hidden=128,
        dropout=0.3,
        physio_dim=physio_dim
    )
    model.eval()
    dummy_mel = torch.zeros(1, 1, target_mel_shape[0], target_mel_shape[1], dtype=torch.float32)
    dummy_mfcc = torch.zeros(1, 1, target_mfcc_shape[0], target_mfcc_shape[1], dtype=torch.float32)
    dummy_physio = torch.zeros(1, physio_dim, dtype=torch.float32)
    with torch.no_grad():
        _ = model(dummy_mel, dummy_mfcc, dummy_physio)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "labels": labels,
        "target_mel_shape": target_mel_shape,
        "target_mfcc_shape": target_mfcc_shape,
        "mel_mean": float(checkpoint.get("mel_mean", 0.0)),
        "mel_std": float(checkpoint.get("mel_std", 1.0)),
        "mfcc_full_mean": float(checkpoint.get("mfcc_full_mean", 0.0)),
        "mfcc_full_std": float(checkpoint.get("mfcc_full_std", 1.0)),
        "physio_mean": checkpoint.get("physio_mean", [0.0, 0.0, 0.0, 0.0]),
        "physio_std": checkpoint.get("physio_std", [1.0, 1.0, 1.0, 1.0]),
        "physio_dim": physio_dim,
        "mode": "multimodal"
    }

# =========================================================
# INFERENCE
# =========================================================
def predict_wav_audio_only(local_wav_path, bundle):
    model = bundle["model"]
    labels = bundle["labels"]
    target_mel_h, target_mel_w = bundle["target_mel_shape"]
    target_mfcc_h, target_mfcc_w = bundle["target_mfcc_shape"]

    sr, x = load_wav(local_wav_path)
    if not is_valid_audio(x, sr):
        raise ValueError("Le fichier audio semble vide, corrompu ou non exploitable.")

    x = normalize_peak(resample_audio(x, sr, TARGET_SR))
    segments = segment_audio(x, SEGMENT_SAMPLES)

    seg_rows = []
    probs_all = []

    with torch.no_grad():
        for seg_id, start, end, chunk in segments:
            mel, mfcc = extract_features(chunk)

            mel = pad_or_trim_2d(mel, target_mel_h, target_mel_w)
            mfcc = pad_or_trim_2d(mfcc, target_mfcc_h, target_mfcc_w)

            mel_std = bundle["mel_std"] if bundle["mel_std"] != 0 else 1.0
            mfcc_std = bundle["mfcc_full_std"] if bundle["mfcc_full_std"] != 0 else 1.0

            mel = ((mel - bundle["mel_mean"]) / mel_std).astype(np.float32)
            mfcc = ((mfcc - bundle["mfcc_full_mean"]) / mfcc_std).astype(np.float32)

            mel_tensor = torch.tensor(mel[None, None, :, :], dtype=torch.float32)
            mfcc_tensor = torch.tensor(mfcc[None, None, :, :], dtype=torch.float32)

            logits, _ = model(mel_tensor, mfcc_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_all.append(probs)

            row = {
                "segment_id": seg_id,
                "start_sec": round(start / TARGET_SR, 2),
                "end_sec": round(end / TARGET_SR, 2),
            }
            for i, label in enumerate(labels):
                row[label] = float(probs[i])
            seg_rows.append(row)

    mean_probs = np.mean(np.stack(probs_all), axis=0)
    result_df = pd.DataFrame({"Condition": labels, "Probability": mean_probs}).sort_values("Probability", ascending=False)

    pred_label = str(result_df.iloc[0]["Condition"])
    confidence = float(result_df.iloc[0]["Probability"])
    return pred_label, confidence, result_df, pd.DataFrame(seg_rows)


def predict_wav_multimodal(local_wav_path, bundle, physio_raw):
    model = bundle["model"]
    labels = bundle["labels"]
    target_mel_h, target_mel_w = bundle["target_mel_shape"]
    target_mfcc_h, target_mfcc_w = bundle["target_mfcc_shape"]

    physio = np.array(physio_raw, dtype=np.float32)
    physio = (
        (physio - np.array(bundle["physio_mean"], dtype=np.float32)) /
        np.array(bundle["physio_std"], dtype=np.float32)
    ).astype(np.float32)

    sr, x = load_wav(local_wav_path)
    if not is_valid_audio(x, sr):
        raise ValueError("Le fichier audio semble vide, corrompu ou non exploitable.")

    x = normalize_peak(resample_audio(x, sr, TARGET_SR))
    segments = segment_audio(x, SEGMENT_SAMPLES)

    seg_rows = []
    probs_all = []

    with torch.no_grad():
        for seg_id, start, end, chunk in segments:
            mel, mfcc = extract_features(chunk)

            mel = pad_or_trim_2d(mel, target_mel_h, target_mel_w)
            mfcc = pad_or_trim_2d(mfcc, target_mfcc_h, target_mfcc_w)

            mel_std = bundle["mel_std"] if bundle["mel_std"] != 0 else 1.0
            mfcc_std = bundle["mfcc_full_std"] if bundle["mfcc_full_std"] != 0 else 1.0

            mel = ((mel - bundle["mel_mean"]) / mel_std).astype(np.float32)
            mfcc = ((mfcc - bundle["mfcc_full_mean"]) / mfcc_std).astype(np.float32)

            mel_tensor = torch.tensor(mel[None, None, :, :], dtype=torch.float32)
            mfcc_tensor = torch.tensor(mfcc[None, None, :, :], dtype=torch.float32)
            physio_tensor = torch.tensor(physio[None, :], dtype=torch.float32)

            logits, _ = model(mel_tensor, mfcc_tensor, physio_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_all.append(probs)

            row = {
                "segment_id": seg_id,
                "start_sec": round(start / TARGET_SR, 2),
                "end_sec": round(end / TARGET_SR, 2),
            }
            for i, label in enumerate(labels):
                row[label] = float(probs[i])
            seg_rows.append(row)

    mean_probs = np.mean(np.stack(probs_all), axis=0)
    result_df = pd.DataFrame({"Condition": labels, "Probability": mean_probs}).sort_values("Probability", ascending=False)

    pred_label = str(result_df.iloc[0]["Condition"])
    confidence = float(result_df.iloc[0]["Probability"])
    return pred_label, confidence, result_df, pd.DataFrame(seg_rows)

# =========================================================
# UI HELPERS
# =========================================================
def format_pct(x):
    return f"{100 * float(x):.2f}%"


def clinical_recommendation(pred_label, confidence, result_df):
    top2 = result_df.head(2).reset_index(drop=True)

    if len(top2) > 1:
        margin = float(top2.loc[0, "Probability"] - top2.loc[1, "Probability"])
    else:
        margin = float(top2.loc[0, "Probability"])

    # Gestion de l'incertitude
    if confidence < 0.55 or margin < 0.10:
        uncertainty = (
            "Le résultat est à interpréter avec prudence, car la confiance est modérée "
            "ou les deux premières classes sont proches."
        )
    else:
        uncertainty = (
            "La séparation entre les classes les plus probables est relativement nette "
            "pour ce fichier."
        )

    # Mapping des interprétations
    mapping = {
        "Healthy": (
            "Le signal est davantage compatible avec une respiration sans anomalie évidente "
            "selon ce modèle. En cas de symptômes persistants, un avis médical reste nécessaire."
        ),
        "Asthma": (
            "Le profil est davantage compatible avec un tableau asthmatique. À confronter aux "
            "symptômes, à l'auscultation et idéalement à une exploration fonctionnelle respiratoire."
        ),
        "Bronchial": (
            "Le profil est davantage compatible avec une atteinte bronchique. Une corrélation "
            "clinique et une auscultation sont recommandées."
        ),
        "Copd": (
            "Le profil est davantage compatible avec une BPCO/COPD. À interpréter avec le "
            "tabagisme, la dyspnée chronique et l'évaluation pneumologique."
        ),
        "Pneumonia": (
            "Le profil est davantage compatible avec une pneumonie. Une confirmation clinique "
            "rapide est conseillée, surtout en cas de fièvre, toux productive ou désaturation."
        )
    }

    common = "Cet outil est une aide à l'orientation et ne remplace pas un diagnostic médical."
    emergency = (
        "Consultez en urgence si détresse respiratoire, cyanose, confusion, douleur thoracique "
        "importante ou aggravation rapide."
    )

    # Retour final
    return (
        f"{mapping.get(pred_label, 'Interprétation clinique non disponible.')}\n\n"
        f"{uncertainty}\n\n"
        f"{common}\n\n"
        f"{emergency}"
    )
def display_kpis(pred_label, confidence, n_segments, second_label, second_prob, mode_used):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Classe prédite", pred_label)
    c2.metric("Confiance", format_pct(confidence))
    c3.metric("Segments analysés", n_segments)
    c4.metric("2e classe", f"{second_label} ({format_pct(second_prob)})")
    c5.metric("Mode", mode_used)


def save_prediction(file_name, mode_used, pred_label, confidence, result_df, segment_df, recommendation):
    if not SAVE_PREDICTIONS:
        return None
    ensure_predictions_table()
    session = get_session()
    pred_id = str(uuid.uuid4())
    payload = [{
        "PREDICTION_ID": pred_id,
        "CREATED_AT": datetime.now(pytz.timezone("Europe/Paris")).strftime('%Y-%m-%d %H:%M:%S'),
        "FILE_NAME": file_name,
        "MODE_USED": mode_used,
        "N_SEGMENTS": int(len(segment_df)),
        "PREDICTED_LABEL": pred_label,
        "CONFIDENCE": float(confidence),
        "PROBABILITIES_JSON": result_df.to_json(orient="records"),
        "SEGMENT_DETAILS_JSON": segment_df.to_json(orient="records"),
        "RECOMMENDATION": recommendation,
    }]
    session.write_pandas(pd.DataFrame(payload), PREDICTIONS_TABLE, auto_create_table=False, overwrite=False)
    return pred_id

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.subheader("Configuration")
    st.write(f"Sampling cible : **{TARGET_SR} Hz**")
    st.write(f"Segment : **{SEGMENT_SECONDS} s**")
    st.info("Les probabilités finales sont calculées par moyenne sur tous les segments de 5 secondes.")
    
    st.markdown("---")

    # 👇 PUSH VERS LE BAS (astuce)
    st.markdown("<br>", unsafe_allow_html=True)
#   st.caption("M2 BIG DATA")

    with st.expander("❓  Guide d'utilisation"):
        st.markdown("**1.** Chargez un fichier `.wav` respiratoire")
        st.markdown("**2.** Activez la fusion multimodale si vous avez des données physiologiques")
        st.markdown("**3.** Cliquez sur **Lancer l'analyse**")
        st.markdown("**4.** Consultez les résultats")
        st.markdown("**5.** Consultez l'historique")
        
    with st.expander("📬   Nous contacter"):
        st.markdown("**Abdelkrim Ghilas**")
        st.markdown("[Data Scientist](https://www.linkedin.com/in/abdelkrim-ghilas-972b70277/)")
        st.markdown("---")
        st.markdown("**Mazigh Kernou**")
        st.markdown("[Data Engineer](https://www.linkedin.com/in/mazigh-kernou/)")
        st.markdown("---")
        st.markdown("**Abdeslem Issaadi**")
        st.markdown("[Marketing Data Analyst](https://www.linkedin.com/in/abdeslem-issaadi-8aa6b0180/)")
        
        
        
# =========================================================
# MAIN UI
# =========================================================
uploaded = st.file_uploader("Chargez un fichier WAV respiratoire", type=["wav"])
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Analyse", "Historique récent", "Dashboard"])

with tab1:
    st.subheader("Paramètres physiologiques")
    use_physio = st.checkbox("Activer la fusion multimodale", value=False)

    spo2 = None
    temperature = None
    systolic_bp = None
    diastolic_bp = None

    if use_physio:
        c1, c2 = st.columns(2)
        with c1:
            spo2 = st.slider("SpO2 (%)", 80, 100, 92)
            temperature = st.slider("Température (°C)", 35.0, 40.0, 37.0, 0.1)
        with c2:
            systolic_bp = st.slider("Tension systolique", 90, 180, 125)
            diastolic_bp = st.slider("Tension diastolique", 60, 120, 80)

    if uploaded is not None:
        st.audio(uploaded, format="audio/wav")

        if st.button("Lancer l'analyse", type="primary"):
            
    
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded.getbuffer())
                    local_path = tmp.name

                if use_physio:
                    bundle = load_multimodal_bundle()
                    physio_raw = [spo2, temperature, systolic_bp, diastolic_bp]
                    mode_used = "multimodal"

                    with st.spinner("Analyse multimodale en cours..."):
                        pred_label, confidence, result_df, segment_df = predict_wav_multimodal(
                            local_path, bundle, physio_raw
                        )
                else:
                    bundle = load_baseline_bundle()
                    mode_used = "audio_only"

                    with st.spinner("Analyse audio en cours..."):
                        pred_label, confidence, result_df, segment_df = predict_wav_audio_only(
                            local_path, bundle
                        )

                try:
                    os.remove(local_path)
                except OSError:
                    pass

                second_label = str(result_df.iloc[1]["Condition"]) if len(result_df) > 1 else "-"
                second_prob = float(result_df.iloc[1]["Probability"]) if len(result_df) > 1 else 0.0
                reco = clinical_recommendation(pred_label, confidence, result_df)
                # ✅ AFFICHAGE PROPRE HTML
                #st.markdown(reco, unsafe_allow_html=True)
                prediction_id = save_prediction(
                    uploaded.name,
                    mode_used,
                    pred_label,
                    confidence,
                    result_df,
                    segment_df,
                    reco
                )

                st.success("Analyse terminée.")
                if prediction_id:
                    st.caption(f"Prediction ID: {prediction_id}")

                display_kpis(
                    pred_label,
                    confidence,
                    len(segment_df),
                    second_label,
                    second_prob,
                    mode_used
                )

                left, right = st.columns([1.2, 1.0])

                with left:
                    st.subheader("Probabilités par condition")
                    chart = alt.Chart(result_df).mark_bar().encode(
                        x=alt.X("Condition:N", sort='-y'),
                        y=alt.Y(
                            "Probability:Q",
                            scale=alt.Scale(domain=[0, 1])  # 🔒 limite entre 0% et 100%
                        ),
                        tooltip=["Condition", "Probability"]
                    ).interactive()

                    st.altair_chart(chart, use_container_width=True)
                    show_df = result_df.copy()
                    show_df["Probability"] = show_df["Probability"].map(format_pct)
                    st.dataframe(show_df, hide_index=True, use_container_width=True)

                with right:
                    st.subheader("Interprétation")
                    st.warning(reco)

                st.subheader("Détail par segment")
                seg_show = segment_df.copy()
                for label in bundle["labels"]:
                    seg_show[label] = seg_show[label].map(format_pct)
                st.dataframe(seg_show, hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur pendant l'analyse : {e}")
    
with tab2:
    try:
        ensure_predictions_table()
        hist = get_session().table(PREDICTIONS_TABLE).sort("CREATED_AT", ascending=False).to_pandas()
        if len(hist) == 0:
            st.info("Aucune prédiction enregistrée pour l'instant.")
        else:
            hist_show = hist.copy()
            if "CONFIDENCE" in hist_show.columns:
                hist_show["CONFIDENCE"] = hist_show["CONFIDENCE"].map(format_pct)
            st.dataframe(hist_show, hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Historique indisponible : {e}")

with tab3:
    st.subheader("Dashboard RespiraSense AI")

    session = get_session()

    try:
        # =========================
        # KPI globaux
        # =========================
        kpi_df = session.sql("""
            SELECT *
            FROM VW_KPI_GLOBAL
        """).to_pandas()

        if not kpi_df.empty:
            k = kpi_df.iloc[0]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total prédictions", int(k["TOTAL_PREDICTIONS"]))
            c2.metric("Aujourd’hui", int(k["PREDICTIONS_TODAY"]))
            c3.metric("Cette semaine", int(k["PREDICTIONS_THIS_WEEK"]))
            c4.metric("Confiance moyenne", f"{100 * float(k['AVG_CONFIDENCE']):.2f}%")
            c5.metric("Taux non Healthy", f"{float(k['NON_HEALTHY_RATE_PCT']):.2f}%")

        st.markdown("---")

        # =========================
        # Répartition des classes
        # =========================
        st.markdown("### Répartition des classes")
        class_df = session.sql("""
            SELECT classe_predite, nb_cases, pct_cases, avg_confidence
            FROM VW_CLASS_DISTRIBUTION
        """).to_pandas()

        if not class_df.empty:
            class_chart = class_df.rename(columns={
                "CLASSE_PREDITE": "Classe",
                "NB_CASES": "Nombre"
            })
            st.bar_chart(class_chart.set_index("Classe")["Nombre"])
            st.dataframe(class_df, use_container_width=True, hide_index=True)


        # =========================
        # Évolution temporelle
        # =========================
        st.markdown("### Évolution temporelle")
        trend_df = session.sql("""
            SELECT prediction_date, classe_predite, nb_cases
            FROM VW_DAILY_TRENDS
            ORDER BY prediction_date
        """).to_pandas()

        if not trend_df.empty:
            trend_df.columns = [c.upper() for c in trend_df.columns]
            pivot_trend = trend_df.pivot_table(
                index="PREDICTION_DATE",
                columns="CLASSE_PREDITE",
                values="NB_CASES",
                aggfunc="sum",
                fill_value=0
            )
            pivot_trend_reset = pivot_trend.reset_index().melt(
                 id_vars="PREDICTION_DATE",
                 var_name="Classe",
                 value_name="Nombre"
            )

            chart = alt.Chart(pivot_trend_reset).mark_line().encode(
                x=alt.X("PREDICTION_DATE:T"),
                y=alt.Y(
                    "Nombre:Q",
                    scale=alt.Scale(domain=[0, pivot_trend_reset["Nombre"].max()])
                ),
                color="Classe:N",
                tooltip=["PREDICTION_DATE", "Classe", "Nombre"]
            ).interactive()

            st.altair_chart(chart, use_container_width=True)


    except Exception as e:
        st.warning(f"Dashboard indisponible : {e}")