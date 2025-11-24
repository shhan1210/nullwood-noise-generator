import os
import io
import numpy as np
from flask import Flask, send_file, request, render_template_string

try:
    from scipy.io.wavfile import write, read as wav_read
except ImportError:
    from scipy.io.wavfile import write  # type: ignore
    wav_read = None  # type: ignore

app = Flask(__name__)

OST_SAMPLE_RATE = None
OST_DATA = None

def load_ost():
    """Ephemeral OST 로딩 (static/Ephemeral.wav)"""
    global OST_SAMPLE_RATE, OST_DATA
    if wav_read is None:
        return
    ost_path = os.path.join(app.root_path, "static", "Ephemeral.wav")
    if not os.path.exists(ost_path):
        print("Ephemeral.wav not found in static folder")
        return
    try:
        sr, data = wav_read(ost_path)
        data = np.asarray(data, dtype=np.float32)
        # 모노 → 스테레오 변환
        if data.ndim == 1:
            maxv = np.max(np.abs(data)) or 1.0
            data = (data / maxv).astype(np.float32)
            data = np.stack([data, data], axis=1)
        else:
            maxv = np.max(np.abs(data)) or 1.0
            data = (data / maxv).astype(np.float32)
        OST_SAMPLE_RATE = sr
        OST_DATA = data
        print("Ephemeral OST loaded.")
    except Exception as e:
        print("Failed to load Ephemeral OST:", e)

def make_loopable(signal: np.ndarray, crossfade_sec: float, sample_rate: int) -> np.ndarray:
    n_samples = signal.shape[0]
    channels = signal.shape[1]
    fade_len = int(sample_rate * crossfade_sec)
    if fade_len <= 0 or fade_len * 2 >= n_samples:
        return signal
    out = signal.copy()
    fade = np.linspace(0.0, 1.0, fade_len)
    for ch in range(channels):
        head = out[:fade_len, ch]
        tail = out[-fade_len:, ch]
        mixed_head = head * (1.0 - fade) + tail * fade
        mixed_tail = head * fade + tail * (1.0 - fade)
        out[:fade_len, ch] = mixed_head
        out[-fade_len:, ch] = mixed_tail
    return out

def generate_nullwood_noise(
    duration_sec=60,
    sample_rate=44100,
    profile="Rain",
    time_intensity=0,
    amp_intensity=0,
    anxiety=0,
    calm=50,
    melody_intensity=0,
    pulse_intensity=0,
    void_intensity=0,
    snow_intensity=0,
    leaf_intensity=0,
    ost_mix=0,
    volume=80
):
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)

    # 기본 화이트 노이즈
    base_noise = np.random.normal(0, 1, n)

    # 프로필 매핑: Rain, Storm
    if profile == "Rain":
        color = "blue"
        base_calm = 0.30
        stereo_move_base = 0.35
        profile_gain = 0.9
    elif profile == "Storm":
        # 폭풍우는 거친 노이즈 (brown)
        color = "brown"
        base_calm = 0.40
        stereo_move_base = 0.20
        profile_gain = 1.0
    else:
        color = "blue"
        base_calm = 0.30
        stereo_move_base = 0.35
        profile_gain = 0.9

    # 색 노이즈 변환
    noise = base_noise.copy()
    if color == "pink":
        noise = np.cumsum(noise)
    elif color == "brown":
        noise = np.cumsum(np.cumsum(noise))
    elif color == "blue":
        noise = np.diff(np.concatenate([[0], noise]))
    noise = noise / (np.max(np.abs(noise)) + 1e-9)

    # 기본 부드러움
    if base_calm > 0:
        window = int(1 + base_calm * 4000)
        kernel = np.ones(window, dtype=np.float32) / float(window)
        blurred = np.convolve(noise, kernel, mode="same")
        noise = (1 - base_calm) * noise + base_calm * blurred

    # 스테레오 패닝
    pan_base = np.sin(2 * np.pi * 0.03 * t) * stereo_move_base
    left = noise * (1 - pan_base)
    right = noise * (1 + pan_base)
    left *= profile_gain
    right *= profile_gain

    # 발자국 (time)
    time_norm = np.clip(float(time_intensity) / 100.0, 0.0, 1.0)
    if time_norm > 0:
        min_rate, max_rate = 0.4, 3.0
        base_rate = min_rate + (max_rate - min_rate) * time_norm
        step = 0
        while step < n:
            interval_sec = np.random.uniform(0.6, 1.4) / base_rate
            interval = int(interval_sec * sample_rate)
            step += interval
            if step >= n: break
            length = int(sample_rate * np.random.uniform(0.06, 0.12))
            start = step; end = min(n, start + length); seg_len = end - start
            if seg_len <= 4: continue
            env = np.linspace(0, 1, seg_len)
            env = env * (1 - env)
            env = env ** 0.7
            low_freq = np.random.uniform(80, 140)
            phase = np.random.uniform(0, 2 * np.pi)
            tone_low = np.sin(2 * np.pi * low_freq * t[start:end] + phase)
            mid_noise = np.random.normal(0, 1, seg_len)
            mid_kernel = np.ones(40, dtype=np.float32) / 40.0
            mid_noise = np.convolve(mid_noise, mid_kernel, mode="same")
            step_wave = (tone_low * 0.7 + mid_noise * 0.6) * env * 0.9
            # 연속 발자국
            if np.random.rand() < 0.4 and end + seg_len < n:
                gap = int(sample_rate * np.random.uniform(0.05, 0.12))
                start2 = end + gap
                end2 = min(n, start2 + seg_len)
                seg_len2 = end2 - start2
                if seg_len2 > 4:
                    env2 = np.linspace(0, 1, seg_len2)
                    env2 = env2 * (1 - env2)
                    env2 = env2 ** 0.7
                    step2 = (np.random.normal(0, 1, seg_len2) * 0.5) * env2 * 0.7
                    pan2 = np.random.uniform(-0.3, 0.3)
                    left[start2:end2] += step2 * (1 - pan2)
                    right[start2:end2] += step2 * (1 + pan2)
            pan = np.random.uniform(-0.3, 0.3)
            left[start:end] += step_wave * (1 - pan)
            right[start:end] += step_wave * (1 + pan)

    # 랜덤 증폭기 (amp modulation)
    amp_norm = np.clip(float(amp_intensity) / 100.0, 0.0, 1.0)
    if amp_norm > 0:
        amp_mod = np.ones(n, dtype=np.float32)
        events_per_sec = 0.2 + amp_norm * 1.0
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            length_sec = np.random.uniform(0.5, 2.0)
            start = int(start_time * sample_rate)
            end = min(n, int((start_time + length_sec) * sample_rate))
            seg_len = end - start
            if seg_len <= sample_rate * 0.2: continue
            # 선택된 세그먼트의 증폭 정도 (0.5~1.5 범위)
            amp_factor = 1.0 + (np.random.uniform(-0.5, 0.5) * amp_norm)
            env = np.linspace(0, 1, seg_len)
            env = np.minimum(env, env[::-1]) ** 1.0
            envelope = 1 + (amp_factor - 1) * env
            amp_mod[start:end] *= envelope
        left *= amp_mod
        right *= amp_mod

    # 새
    bird_norm = np.clip(float(anxiety) / 100.0, 0.0, 1.0)
    if bird_norm > 0:
        events_per_sec = 0.1 + bird_norm * 1.5
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            start = int(start_time * sample_rate)
            length = int(sample_rate * np.random.uniform(0.08, 0.25))
            end = min(n, start + length)
            seg_len = end - start
            if seg_len <= 4: continue
            env = np.hanning(seg_len)
            base_freq = np.random.uniform(2200, 4200)
            vibrato = np.sin(2 * np.pi * np.linspace(0, 1, seg_len) * np.random.uniform(6, 11))
            phase = np.cumsum(2 * np.pi * (base_freq + base_freq * 0.08 * vibrato) / sample_rate)
            chirp = np.sin(phase) * env * 0.5
            pan = np.random.uniform(-0.8, 0.8)
            left[start:end] += chirp * (1 - pan)
            right[start:end] += chirp * (1 + pan)

    # 멜로디 (far bell)
    melody_norm = np.clip(float(melody_intensity) / 100.0, 0.0, 1.0)
    if melody_norm > 0:
        events_per_sec = 0.5 + melody_norm * 2.5
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            length_sec = np.random.uniform(0.4, 1.2)
            start = int(start_time * sample_rate)
            end = min(n, int((start_time + length_sec) * sample_rate))
            seg_len = end - start
            if seg_len <= 4: continue
            env = np.linspace(0, 1, seg_len)
            env = env[::-1] ** 2.5
            part_count = np.random.randint(2, 4)
            tone = np.zeros(seg_len, dtype=np.float32)
            for _i in range(part_count):
                freq = np.random.uniform(200, 500)
                phase = 2 * np.pi * freq * t[start:end]
                tone += np.sin(phase)
            tone = tone / part_count
            bell = tone * env * (0.15 + 0.25 * melody_norm)
            pan = np.random.uniform(-0.5, 0.5)
            left[start:end] += bell * (1 - pan)
            right[start:end] += bell * (1 + pan)

    # 미스테리
    pulse_norm = np.clip(float(pulse_intensity) / 100.0, 0.0, 1.0)
    if pulse_norm > 0:
        events_per_sec = 0.2 + pulse_norm * 1.0
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            start = int(start_time * sample_rate)
            length = int(sample_rate * np.random.uniform(0.3, 0.8))
            end = min(n, start + length)
            seg_len = end - start
            if seg_len <= 8: continue
            env = np.linspace(0, 1, seg_len)
            env = np.minimum(env, env[::-1]) ** 1.5
            band = np.random.uniform(200, 600)
            rough = np.random.normal(0, 1, seg_len)
            phase = 2 * np.pi * band * np.arange(seg_len) / sample_rate
            rustle = (0.6 * np.sin(phase) + 0.4 * rough) * env * 0.5
            pan = np.random.uniform(-0.6, 0.6)
            left[start:end] += rustle * (1 - pan)
            right[start:end] += rustle * (1 + pan)

    # 방해 (void)
    void_norm = np.clip(float(void_intensity) / 100.0, 0.0, 1.0)
    if void_norm > 0:
        max_rate = 0.8
        events_per_sec = 0.1 + void_norm * max_rate
        num_voids = max(1, int(events_per_sec * duration_sec))
        for _ in range(num_voids):
            start_time = np.random.uniform(0, duration_sec)
            void_len = np.random.uniform(0.08, 0.35)
            start = int(start_time * sample_rate)
            end = min(n, int((start_time + void_len) * sample_rate))
            seg_len = end - start
            if seg_len <= 4: continue
            fade_len = min(int(sample_rate * 0.02), seg_len // 2)
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            left[start:start + fade_len] *= fade
            right[start:start + fade_len] *= fade
            left[end - fade_len:end] *= fade[::-1]
            right[end - fade_len:end] *= fade[::-1]
            if end - start - 2 * fade_len > 0:
                left[start + fade_len:end - fade_len] *= 0.1
                right[start + fade_len:end - fade_len] *= 0.1

    # 스노우 (snow)
    snow_norm = np.clip(float(snow_intensity) / 100.0, 0.0, 1.0)
    if snow_norm > 0:
        events_per_sec = 0.5 + snow_norm * 1.5
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            length = int(sample_rate * np.random.uniform(0.05, 0.25))
            start = int(start_time * sample_rate)
            end = min(n, start + length)
            seg_len = end - start
            if seg_len <= 4: continue
            env = np.linspace(0, 1, seg_len)
            env = np.minimum(env, env[::-1]) ** 0.8
            noise_seg = np.random.normal(0, 1, seg_len)
            # 고주파 강조
            blur = np.convolve(noise_seg, np.ones(20, dtype=np.float32) / 20.0, mode="same")
            snow_chunk = (noise_seg - blur) * env * (0.4 + 0.6 * snow_norm)
            pan = np.random.uniform(-0.4, 0.4)
            left[start:end] += snow_chunk * (1 - pan)
            right[start:end] += snow_chunk * (1 + pan)

    # 낙옆 (leaf)
    leaf_norm = np.clip(float(leaf_intensity) / 100.0, 0.0, 1.0)
    if leaf_norm > 0:
        events_per_sec = 0.2 + leaf_norm * 1.2
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            start = int(start_time * sample_rate)
            length = int(sample_rate * np.random.uniform(0.1, 0.35))
            end = min(n, start + length)
            seg_len = end - start
            if seg_len <= 4: continue
            env = np.linspace(0, 1, seg_len)
            env = np.minimum(env, env[::-1]) ** 0.7
            noise_seg = np.random.normal(0, 1, seg_len)
            smooth = np.convolve(noise_seg, np.ones(20, dtype=np.float32)/20.0, mode="same")
            crunch = (noise_seg - smooth) * env * (0.5 + 0.5 * leaf_norm)
            pan = np.random.uniform(-0.4, 0.4)
            left[start:end] += crunch * (1 - pan)
            right[start:end] += crunch * (1 + pan)

    # 기본 방향 조정
    dir_norm = np.clip((float(calm) - 50.0) / 50.0, -1.0, 1.0)
    if abs(dir_norm) > 0.01:
        if dir_norm > 0:
            left *= 1.0 - dir_norm
        else:
            right *= 1.0 + dir_norm

    stereo = np.vstack([left, right]).T
    noise_only = stereo.copy()

    # OST 믹스 (Ephemeral)
    ost_norm = np.clip(float(ost_mix) / 100.0, 0.0, 1.0)
    if OST_DATA is not None and ost_norm > 0:
        ratio = OST_SAMPLE_RATE / float(sample_rate) if OST_SAMPLE_RATE else 1.0
        idx = (np.arange(n) * ratio).astype(int) % OST_DATA.shape[0]
        ost_segment = OST_DATA[idx, :]
        noise_weight = 1.0 - 0.95 * ost_norm
        ost_weight = ost_norm
        stereo = noise_only * noise_weight + ost_segment * ost_weight
    else:
        stereo = noise_only

    # 정규화 및 볼륨
    maxv = np.max(np.abs(stereo)) or 1.0
    stereo = stereo / maxv
    volume_norm = np.clip(float(volume) / 100.0, 0.0, 1.0)
    stereo *= volume_norm

    # 루프 페이드
    crossfade_sec = 0.0 if ost_norm > 0 else 1.0
    stereo = make_loopable(stereo, crossfade_sec=crossfade_sec, sample_rate=sample_rate)
    stereo_int16 = (stereo * 32767).astype(np.int16)
    buf = io.BytesIO()
    write(buf, sample_rate, stereo_int16)
    buf.seek(0)
    filename = f"nullwood_{profile}_{duration_sec}s.wav"
    return buf, filename

# ===================================
# Web UI
# ===================================

@app.route("/")
def index():
    lang = "ko"
    html = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title data-i18n="title">Nullwood 노이즈 생성기</title>
    <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #000;
        color: #f4f4f4;
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        padding-top: 40px;
        background-image: url("/static/nullwood_bg.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .panel {
        width: 420px;
        max-width: 95vw;
        background: rgba(7,8,11,0.8);
        border-radius: 18px;
        padding: 20px 22px 18px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.8);
        backdrop-filter: blur(10px);
    }
    .title { font-size: 20px; font-weight: 600; letter-spacing: 0.06em; text-align: center; margin-bottom: 4px; }
    .subtitle { font-size: 12px; text-align: center; color:#c0c4d0; margin-bottom:12px; }
    .lang-switch { display:flex; justify-content:center; gap:8px; margin-bottom:12px; }
    .lang-btn { border:0; padding:2px 8px; font-size:11px; border-radius:999px; background:transparent; color:#9ca3af; cursor:pointer; }
    .lang-btn.active { background:#2563eb; color:#f9fafb; }
    .eq-bar { width:100%; height:64px; background:rgba(15,23,42,0.6); border-radius:12px; overflow:hidden; margin-bottom:8px; }
    #eqCanvas { width:100%; height:100%; display:block; }
    .noise-toggle-row { margin-bottom:4px; }
    .noise-btn { width:100%; border:0; padding:6px 10px; border-radius:999px; font-size:11px; cursor:pointer; background:#111827; color:#e5e7eb; }
    .noise-btn.active { background:#dc2626; color:#fee2e2; }
    .tips { margin-top:4px; font-size:10px; color:#9ca3af; line-height:1.4; margin-bottom:8px; }
    .section-title { font-size:12px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-top:10px; margin-bottom:4px; }
    label { display:block; font-size:12px; color:#e5e7eb; margin-top:8px; margin-bottom:2px; }
    select, input[type="number"], input[type="text"] {
        width:100%; padding:6px 8px; border-radius:8px;
        border:1px solid #374151; background:rgba(15,23,42,0.9);
        color:#e5e7eb; font-size:12px; outline:none;
    }
    select:focus, input[type="number"]:focus, input[type="text"]:focus { border-color:#2563eb; }
    .slider-row { display:flex; align-items:center; gap:8px; }
    .slider-row span.value { width:34px; text-align:right; font-size:11px; color:#9ca3af; }
    input[type="range"] { flex:1; -webkit-appearance:none; height:4px; border-radius:999px; background:#111827; outline:none; }
    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance:none; width:14px; height:14px;
        border-radius:50%; background:#f9fafb; border:1px solid #4b5563; cursor:pointer; margin-top:-5px;
    }
    .download-row { margin-top:8px; display:flex; gap:8px; }
    .download-row input[type="number"] { width:70px; }
    .download-btn { flex:1; border:0; border-radius:10px; font-size:12px; padding:7px 10px; background:#2563eb; color:#f9fafb; cursor:pointer; white-space:nowrap; }
    .download-btn:hover { background:#1d4ed8; }
    .player-row { margin-top:10px; display:flex; align-items:center; gap:8px; }
    .player-row audio { width:100%; }
    .ost-links { margin-top:4px; font-size:11px; color:#e5e7eb; }
    .ost-links a { color:#93c5fd; text-decoration:none; margin-right:10px; }
    .ost-links a:hover { text-decoration:underline; }
    .footer { margin-top:10px; text-align:center; }
    .footer-logo img { height:40px; opacity:0.85; } /* 로고 2배 확대 */
    .footer-links { margin-top:4px; font-size:10px; color:#9ca3af; line-height:1.5; }
    .footer-links a { color:#bfdbfe; text-decoration:none; margin:0 4px; }
    .footer-links a:hover { text-decoration:underline; }
    .footer-license { margin-top:4px; font-size:9px; color:#6b7280; line-height:1.4; }
    </style>
</head>
<body>
<div class="panel">
    <div class="title" data-i18n="title">Nullwood 노이즈 생성기</div>
    <div class="subtitle" data-i18n="subtitle">Nullwood 노이즈 재생기는 SHHAN의 Nullwood 세계관을 기반으로 한 백색소음 생성기입니다.</div>

    <div class="lang-switch">
        <button class="lang-btn" data-lang="en">EN</button>
        <button class="lang-btn active" data-lang="ko">한국어</button>
        <button class="lang-btn" data-lang="ja">日本語</button>
        <button class="lang-btn" data-lang="zh">繁體中文</button>
    </div>

    <div class="eq-bar"><canvas id="eqCanvas"></canvas></div>
    <div class="noise-toggle-row">
        <button class="noise-btn" id="noiseToggleBtn" data-i18n="noiseOn">노이즈 켜기</button>
    </div>
    <div class="player-row">
        <audio id="audioPlayer" controls loop></audio>
    </div>
    <div class="tips" data-i18n="tips">
        브라우저 정책으로 인해 처음에는 위의 ‘노이즈 켜기’ 버튼을 누른 뒤,
        플레이 버튼을 한 번 눌러야 할 수 있습니다. 슬라이더를 조정하면 현재 설정으로 자동으로 새 소리가 생성됩니다.
    </div>

    <div class="section-title" data-i18n="sceneTitle">시간대별 노이즈</div>
    <select id="profileSelect">
        <option value="Rain">비 (Rain)</option>
        <option value="Storm">폭풍우 (Storm)</option>
    </select>

    <label for="volumeRange" data-i18n="volumeLabel" style="margin-top:10px;">볼륨</label>
    <div class="slider-row">
        <input type="range" id="volumeRange" min="0" max="100" value="56">
        <span class="value" id="volumeValue">56</span>
    </div>

    <div class="section-title" data-i18n="coreTitle">옵션</div>

    <label for="timeRange" data-i18n="timeLabel">발자국</label>
    <div class="slider-row">
        <input type="range" id="timeRange" min="0" max="100" value="0">
        <span class="value" id="timeValue">0</span>
    </div>

    <label for="snowRange" data-i18n="particleLabel">스노우</label>
    <div class="slider-row">
        <input type="range" id="snowRange" min="0" max="100" value="0">
        <span class="value" id="snowValue">0</span>
    </div>

    <label for="windRange" data-i18n="windLabel">랜덤 증폭기</label>
    <div class="slider-row">
        <input type="range" id="windRange" min="0" max="100" value="0">
        <span class="value" id="windValue">0</span>
    </div>

    <label for="anxietyRange" data-i18n="anxietyLabel">새</label>
    <div class="slider-row">
        <input type="range" id="anxietyRange" min="0" max="100" value="0">
        <span class="value" id="anxietyValue">0</span>
    </div>

    <label for="calmRange" data-i18n="calmLabel">방향</label>
    <div class="slider-row">
        <input type="range" id="calmRange" min="0" max="100" value="50">
        <span class="value" id="calmValue">50</span>
    </div>

    <label for="bellRange" data-i18n="bellLabel">멜로디</label>
    <div class="slider-row">
        <input type="range" id="bellRange" min="0" max="100" value="0">
        <span class="value" id="bellValue">0</span>
    </div>

    <label for="pulseRange" data-i18n="pulseLabel">미스테리</label>
    <div class="slider-row">
        <input type="range" id="pulseRange" min="0" max="100" value="0">
        <span class="value" id="pulseValue">0</span>
    </div>

    <label for="voidRange" data-i18n="voidLabel">방해</label>
    <div class="slider-row">
        <input type="range" id="voidRange" min="0" max="100" value="0">
        <span class="value" id="voidValue">0</span>
    </div>

    <label for="leafRange" data-i18n="leafLabel">낙옆</label>
    <div class="slider-row">
        <input type="range" id="leafRange" min="0" max="100" value="0">
        <span class="value" id="leafValue">0</span>
    </div>

    <label for="ostRange" data-i18n="ostLabel">Ephemeral OST 추가</label>
    <div class="slider-row">
        <input type="range" id="ostRange" min="0" max="100" value="0">
        <span class="value" id="ostValue">0</span>
    </div>

    <div class="ost-links">
        <span data-i18n="ostLinksLabel">Ephemeral OST 듣기:</span><br>
        <a href="https://youtu.be/BKf4ZT_SiHM?si=dB4XZZfARuK4lcqX" target="_blank">YouTube</a>
        <a href="https://open.spotify.com/track/???some_id" target="_blank">Spotify</a>
    </div>

    <div class="section-title" style="margin-top:14px;" data-i18n="downloadTitle">다운로드</div>
    <label for="downloadMinutes" data-i18n="downloadLengthLabel">다운로드 길이 (분)</label>
    <div class="download-row">
        <input type="number" id="downloadMinutes" min="1" max="60" value="1">
        <button class="download-btn" id="downloadBtn" data-i18n="downloadBtn">
            현재 옵션으로 널우드 노이즈 다운로드
        </button>
    </div>

    <div class="footer">
        <div class="footer-logo"><img src="/static/nullwood_logo.png" alt="Nullwood logo"></div>
        <div class="footer-links">
            <div>© 2025 SHHAN. All rights reserved.</div>
            <div>
                <a href="https://linktr.ee/shhan1211" target="_blank">Linktree</a> ·
                <a href="https://www.instagram.com/shhan1211" target="_blank">Instagram</a> ·
                <a href="https://www.youtube.com/@shhan1211" target="_blank">YouTube</a> ·
                <a href="https://x.com/shhan1211" target="_blank">X</a> ·
                <a href="https://www.patreon.com/nullwood" target="_blank">Patreon</a>
            </div>
        </div>
        <div class="footer-license">
            Generated audio may be used for personal and commercial projects
            as long as you credit: “Audio by SHHAN – Nullwood Noise Generator”.
            Redistribution of the raw audio alone is not allowed.
        </div>
    </div>
</div>

<script>
// i18n dictionary
const i18n = {
    en: {
        title: "Nullwood Noise Generator",
        subtitle: "The Nullwood noise generator is a white noise generator based on SHHAN's Nullwood universe.",
        sceneTitle: "Time-based noise",
        profileLabel: "Base noise by time of day",
        volumeLabel: "Volume",
        muteBtn: "Mute",
        coreTitle: "Options",
        timeLabel: "Footsteps",
        windLabel: "Random amplifier",
        anxietyLabel: "Birds",
        calmLabel: "Direction",
        bellLabel: "Melody",
        particleLabel: "Snow",
        leafLabel: "Leaves",
        lakeLabel: "Lake",
        catLabel: "Cat",
        pulseLabel: "Mystery",
        voidLabel: "Interference",
        ostLabel: "Add Ephemeral OST",
        ostLinksLabel: "Listen to Ephemeral OST:",
        downloadTitle: "Download",
        downloadLengthLabel: "Download length (minutes)",
        downloadBtn: "Download with current settings",
        tips: "Because of browser policies you may need to click “Turn noise on” and then press play once. Moving any slider will regenerate the sound automatically.",
        noiseOn: "Turn noise on",
        noiseOff: "Turn noise off"
    },
    ko: {
        title: "Nullwood 노이즈 생성기",
        subtitle: "Nullwood 노이즈 재생기는 SHHAN의 Nullwood 세계관을 기반으로 한 백색소음 생성기입니다.",
        sceneTitle: "시간대별 노이즈",
        profileLabel: "시간대별 기본 노이즈",
        volumeLabel: "볼륨",
        muteBtn: "소리 끄기",
        coreTitle: "옵션",
        timeLabel: "발자국",
        windLabel: "랜덤 증폭기",
        anxietyLabel: "새",
        calmLabel: "방향",
        bellLabel: "멜로디",
        particleLabel: "스노우",
        leafLabel: "낙옆",
        lakeLabel: "호수",
        catLabel: "고양이",
        pulseLabel: "미스테리",
        voidLabel: "방해",
        ostLabel: "Ephemeral OST 추가",
        ostLinksLabel: "Ephemeral OST 듣기:",
        downloadTitle: "다운로드",
        downloadLengthLabel: "다운로드 길이 (분)",
        downloadBtn: "현재 옵션으로 널우드 노이즈 다운로드",
        tips: "브라우저 정책으로 인해 처음에는 위의 ‘노이즈 켜기’ 버튼을 누른 뒤, 플레이 버튼을 한 번 눌러야 할 수 있습니다. 슬라이더를 조정하면 현재 설정으로 자동으로 새 소리가 생성됩니다.",
        noiseOn: "노이즈 켜기",
        noiseOff: "노이즈 끄기"
    },
    ja: {
        title: "Nullwood ノイズジェネレーター",
        subtitle: "Nullwoodノイズ再生機はSHHANのNullwood世界観に基づくホワイトノイズ生成機です。",
        sceneTitle: "時間帯ノイズ",
        profileLabel: "時間帯ごとの基本ノイズ",
        volumeLabel: "音量",
        muteBtn: "ミュート",
        coreTitle: "オプション",
        timeLabel: "足音",
        windLabel: "ランダムアンプ",
        anxietyLabel: "鳥",
        calmLabel: "方向",
        bellLabel: "メロディ",
        particleLabel: "スノー",
        leafLabel: "落葉",
        lakeLabel: "湖",
        catLabel: "猫",
        pulseLabel: "ミステリー",
        voidLabel: "妨害",
        ostLabel: "Ephemeral OST を追加",
        ostLinksLabel: "Ephemeral OST を聴く:",
        downloadTitle: "ダウンロード",
        downloadLengthLabel: "ダウンロード時間 (分)",
        downloadBtn: "現在の設定でダウンロード",
        tips: "ブラウザの仕様により、最初は「ノイズを再生」ボタンを押してから、再生ボタンを一度押す必要があるかもしれません。スライダーを動かすと自動的に新しいサウンドが生成されます。",
        noiseOn: "ノイズを再生",
        noiseOff: "ノイズを停止"
    },
    zh: {
        title: "Nullwood 噪音產生器",
        subtitle: "Nullwood噪音播放器是以SHHAN的Nullwood世界觀為基礎的白噪音產生器。",
        sceneTitle: "時間段噪音",
        profileLabel: "依時間區分的基礎噪音",
        volumeLabel: "音量",
        muteBtn: "靜音",
        coreTitle: "選項",
        timeLabel: "腳步",
        windLabel: "隨機放大器",
        anxietyLabel: "鳥",
        calmLabel: "方向",
        bellLabel: "旋律",
        particleLabel: "雪",
        leafLabel: "落葉",
        lakeLabel: "湖泊",
        catLabel: "貓",
        pulseLabel: "謎樣聲音",
        voidLabel: "干擾",
        ostLabel: "加入 Ephemeral OST",
        ostLinksLabel: "收聽 Ephemeral OST:",
        downloadTitle: "下載",
        downloadLengthLabel: "下載長度 (分鐘)",
        downloadBtn: "以目前設定下載",
        tips: "由於瀏覽器政策，第一次可能需要先按下「開啟噪音」按鈕，再按一次播放。移動任何滑桿時會自動產生新的聲音。",
        noiseOn: "開啟噪音",
        noiseOff: "關閉噪音"
    }
};

let currentLang = "{{ lang }}";
let isNoiseOn = false;
let currentParams = null;

function applyLanguage(lang) {
    currentLang = lang;
    const dict = i18n[lang] || i18n["ko"];
    document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.getAttribute("data-i18n");
        if (dict[key]) el.textContent = dict[key];
    });
    const noiseBtn = document.getElementById("noiseToggleBtn");
    if (noiseBtn) {
        noiseBtn.textContent = isNoiseOn ?
            (dict.noiseOff || "노이즈 끄기") :
            (dict.noiseOn || "노이즈 켜기");
        noiseBtn.classList.toggle("active", isNoiseOn);
    }
}

function setupEQ() {
    const canvas = document.getElementById("eqCanvas");
    const ctx = canvas.getContext("2d");
    function resize() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
    }
    resize();
    window.addEventListener("resize", resize);
    let t = 0;
    function draw() {
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        let energy = 0;
        if (isNoiseOn && currentParams) {
            const vol = Number(currentParams.volume || 0) / 100;
            const detail = (
                Number(currentParams.time || 0) +
                Number(currentParams.wind || 0) +
                Number(currentParams.anxiety || 0) +
                Number(currentParams.bell || 0) +
                Number(currentParams.pulse || 0) +
                Number(currentParams.void || 0) +
                Number(currentParams.particle || 0) +
                Number(currentParams.leaf || 0) +
                Number(currentParams.ost || 0)
            ) / (9 * 100);
            energy = 0.15 + vol * 0.6 + detail * 0.4;
            energy = Math.min(energy, 0.8);
        }
        const bars = 48, barWidth = w / bars;
        for (let i=0; i<bars; i++) {
            const phase = t * 0.07 + i * 0.33;
            const baseWave = (Math.sin(phase) + Math.sin(phase * 1.9)) * 0.25 + 0.5;
            const jitter = (Math.random() - 0.5) * 0.4;
            const v = Math.max(0, Math.min(1, baseWave + jitter));
            const bandFactor = 0.6 + 0.4 * Math.sin(i / bars * Math.PI);
            const barH = v * h * 0.5 * energy * bandFactor;
            const x = i * barWidth, y = h - barH;
            ctx.fillStyle = "rgba(255,255,255,0.9)";
            ctx.fillRect(x, y, barWidth * 0.6, barH);
        }
        t += 1;
        requestAnimationFrame(draw);
    }
    draw();
}

function getParams() {
    return {
        profile: document.getElementById("profileSelect").value,
        time: document.getElementById("timeRange").value,
        wind: document.getElementById("windRange").value,
        anxiety: document.getElementById("anxietyRange").value,
        calm: document.getElementById("calmRange").value,
        bell: document.getElementById("bellRange").value,
        pulse: document.getElementById("pulseRange").value,
        void: document.getElementById("voidRange").value,
        particle: document.getElementById("snowRange").value,
        leaf: document.getElementById("leafRange").value,
        ost: document.getElementById("ostRange").value,
        volume: document.getElementById("volumeRange").value
    };
}

let updateTimer = null;

function updateAudio() {
    if (!isNoiseOn) return;
    const audio = document.getElementById("audioPlayer");
    const p = getParams();
    currentParams = p;
    const url = new URL("/stream", window.location.origin);
    Object.entries(p).forEach(([k, v]) => url.searchParams.set(k, v));
    audio.src = url.toString();
    audio.play().catch(() => {});
}

function scheduleUpdate() {
    if (!isNoiseOn) return;
    if (updateTimer) clearTimeout(updateTimer);
    updateTimer = setTimeout(updateAudio, 250);
}

function setupControls() {
    const ids = [
        "volumeRange","timeRange","windRange","anxietyRange","calmRange",
        "bellRange","pulseRange","voidRange",
        "snowRange","leafRange","ostRange","profileSelect"
    ];
    ids.forEach(id => {
        const el = document.getElementById(id);
        const valSpan = document.getElementById(id.replace("Range","Value"));
        if (el && el.type === "range") {
            el.addEventListener("input", () => {
                if (valSpan) valSpan.textContent = el.value;
                scheduleUpdate();
            });
        } else if (el && el.tagName === "SELECT") {
            el.addEventListener("change", () => scheduleUpdate());
        }
    });
    document.getElementById("volumeRange").addEventListener("input", e => {
        document.getElementById("volumeValue").textContent = e.target.value;
    });
    const noiseBtn = document.getElementById("noiseToggleBtn");
    const audio = document.getElementById("audioPlayer");
    noiseBtn.addEventListener("click", () => {
        isNoiseOn = !isNoiseOn;
        const dict = i18n[currentLang] || i18n["ko"];
        noiseBtn.textContent = isNoiseOn ?
            (dict.noiseOff || "노이즈 끄기") :
            (dict.noiseOn || "노이즈 켜기");
        noiseBtn.classList.toggle("active", isNoiseOn);
        if (isNoiseOn) {
            updateAudio();
        } else {
            audio.pause();
            audio.currentTime = 0;
        }
    });
    document.getElementById("downloadBtn").addEventListener("click", () => {
        const minutes = parseInt(document.getElementById("downloadMinutes").value || "1", 10);
        const clamped = Math.min(60, Math.max(1, minutes));
        const p = getParams();
        const url = new URL("/download", window.location.origin);
        Object.entries(p).forEach(([k, v]) => url.searchParams.set(k, v));
        url.searchParams.set("minutes", clamped.toString());
        window.location.href = url.toString();
    });
    document.querySelectorAll(".lang-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".lang-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            applyLanguage(btn.dataset.lang);
        });
    });
}

document.addEventListener("DOMContentLoaded", () => {
    setupEQ();
    applyLanguage("ko");
    setupControls();
});
</script>

</body>
</html>
    """
    return render_template_string(html, lang=lang)

@app.route("/stream")
def stream():
    profile = request.args.get("profile", "Rain")
    time_int = int(request.args.get("time", 0))
    amp = int(request.args.get("wind", 0))
    anxiety = int(request.args.get("anxiety", 0))
    calm = int(request.args.get("calm", 50))
    melody = int(request.args.get("bell", 0))
    pulse = int(request.args.get("pulse", 0))
    void = int(request.args.get("void", 0))
    snow = int(request.args.get("particle", 0))
    leaf = int(request.args.get("leaf", 0))
    ost = int(request.args.get("ost", 0))
    volume = int(request.args.get("volume", 56))
    buf, filename = generate_nullwood_noise(
        duration_sec=60,
        profile=profile,
        time_intensity=time_int,
        amp_intensity=amp,
        anxiety=anxiety,
        calm=calm,
        melody_intensity=melody,
        pulse_intensity=pulse,
        void_intensity=void,
        snow_intensity=snow,
        leaf_intensity=leaf,
        ost_mix=ost,
        volume=volume,
    )
    return send_file(buf, mimetype="audio/wav")

@app.route("/download")
def download():
    profile = request.args.get("profile", "Rain")
    time_int = int(request.args.get("time", 0))
    amp = int(request.args.get("wind", 0))
    anxiety = int(request.args.get("anxiety", 0))
    calm = int(request.args.get("calm", 50))
    melody = int(request.args.get("bell", 0))
    pulse = int(request.args.get("pulse", 0))
    void = int(request.args.get("void", 0))
    snow = int(request.args.get("particle", 0))
    leaf = int(request.args.get("leaf", 0))
    ost = int(request.args.get("ost", 0))
    volume = int(request.args.get("volume", 56))
    minutes = int(request.args.get("minutes", 1))
    minutes = max(1, min(60, minutes))
    duration_sec = minutes * 60
    buf, filename = generate_nullwood_noise(
        duration_sec=duration_sec,
        profile=profile,
        time_intensity=time_int,
        amp_intensity=amp,
        anxiety=anxiety,
        calm=calm,
        melody_intensity=melody,
        pulse_intensity=pulse,
        void_intensity=void,
        snow_intensity=snow,
        leaf_intensity=leaf,
        ost_mix=ost,
        volume=volume,
    )
    download_name = f"{filename[:-4]}_{minutes}min.wav"
    return send_file(buf, as_attachment=True, download_name=download_name, mimetype="audio/wav")

if __name__ == "__main__":
    load_ost()
    app.run(host="127.0.0.1", port=5000, debug=True)