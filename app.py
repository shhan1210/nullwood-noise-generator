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
        print("scipy.io.wavfile 을 사용할 수 없어 OST 믹스를 비활성화합니다.")
        return
    ost_path = os.path.join(app.root_path, "static", "Ephemeral.wav")
    if not os.path.exists(ost_path):
        print("Ephemeral.wav not found in static folder. "
              "Ephemeral.mp4 는 서버 믹스에 사용할 수 없으니, "
              "오디오를 추출해 Ephemeral.wav 로 저장해 주세요.")
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
        print("Ephemeral OST loaded from Ephemeral.wav.")
    except Exception as e:
        print("Failed to load Ephemeral OST:", e)
        OST_SAMPLE_RATE = None
        OST_DATA = None


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
    time_intensity=0,    # 발자국
    amp_intensity=0,     # 울림 (예전 동굴)
    anxiety=0,           # 새
    calm=50,
    melody_intensity=0,  # 멜로디
    pulse_intensity=0,   # 미스테리
    weird_intensity=0,   # 기괴함
    wave_intensity=0,    # 파도
    wind2_intensity=0,   # 바람
    void_intensity=0,    # 방해
    snow_intensity=0,    # 빗자루 (예전 청소)
    electro_intensity=0, # 일렉트로닉
    leaf_intensity=0,    # 현재 사용 안 함
    ost_mix=0,
    volume=80
):
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)

    # 기본 화이트 노이즈
    base_noise = np.random.normal(0, 1, n)

    # 프로필 매핑: Rain, Storm
    if profile == "Storm":
        color = "brown"
        base_calm = 0.20
        stereo_move_base = 0.25
        profile_gain = 1.1
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

    # Storm 전용 추가 질감
    if profile == "Storm":
        rumble_freq = np.random.uniform(20, 45)
        rumble_phase = 2 * np.pi * rumble_freq * t
        rumble = np.sin(rumble_phase)

        lfo_points = max(4, int(duration_sec * 2))
        lfo_x = np.linspace(0, duration_sec, lfo_points)
        lfo_y = np.random.uniform(-1, 1, lfo_points)
        lfo = np.interp(t, lfo_x, lfo_y)
        lfo_env = (np.abs(lfo) ** 1.5)

        rumble *= (0.4 + 0.8 * lfo_env)
        left += rumble * 0.7
        right += rumble * 0.7

        gust = 1.0 + 0.5 * np.maximum(lfo, 0)
        left *= gust
        right *= gust

    # ========================
    # 발자국 (눈밟는 느낌)
    # ========================
    time_norm = np.clip(float(time_intensity) / 100.0, 0.0, 1.0)
    if time_norm > 0:
        min_rate, max_rate = 0.2, 1.0
        base_rate = min_rate + (max_rate - min_rate) * time_norm
        step = 0
        while step < n:
            interval_sec = np.random.uniform(1.2, 2.4) / base_rate
            interval = int(interval_sec * sample_rate)
            step += interval
            if step >= n:
                break

            length = int(sample_rate * np.random.uniform(0.04, 0.09))
            for _ in range(2):  # 두 발자국
                start = step
                end = min(n, start + length)
                seg_len = end - start
                if seg_len > 8:
                    env = np.linspace(1.0, 0.0, seg_len) ** 2.0

                    # 눈 밟는 듯한 고주파 섞인 소리
                    noise_seg = np.random.normal(0, 1, seg_len)
                    blur = np.convolve(
                        noise_seg,
                        np.ones(30, dtype=np.float32) / 30.0,
                        mode="same",
                    )
                    # 하이패스 느낌
                    crunch = (noise_seg - blur)
                    crunch = crunch / (np.max(np.abs(crunch)) + 1e-9)
                    crunch *= env * 0.9

                    # 살짝 저음 섞기 (너무 북 같지 않게 작은 비율)
                    low_freq = np.random.uniform(60, 90)
                    phase = np.random.uniform(0, 2 * np.pi)
                    low = np.sin(2 * np.pi * low_freq * t[start:end] + phase)
                    low *= env * 0.25

                    step_wave = crunch + low
                    pan = np.random.uniform(-0.3, 0.3)
                    left[start:end] += step_wave * (1 - pan)
                    right[start:end] += step_wave * (1 + pan)

                gap_sec = np.random.uniform(0.12, 0.35)
                step += int(gap_sec * sample_rate)
                if step >= n:
                    break

    # ========================
    # 울림 (예전 동굴, 노이즈 변화)
    # ========================
    amp_norm = np.clip(float(amp_intensity) / 100.0, 0.0, 1.0)
    if amp_norm > 0:
        amp_mod = np.ones(n, dtype=np.float32)
        events_per_sec = 0.3 + amp_norm * 1.3
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            length_sec = np.random.uniform(0.8, 2.5)
            start = int(start_time * sample_rate)
            end = min(n, int((start_time + length_sec) * sample_rate))
            seg_len = end - start
            if seg_len <= int(sample_rate * 0.3):
                continue
            amp_factor = 1.0 + (np.random.uniform(-0.7, 0.9) * amp_norm)
            env = np.linspace(0, 1, seg_len)
            env = np.minimum(env, env[::-1])
            envelope = 1 + (amp_factor - 1) * env
            amp_mod[start:end] *= envelope
        left *= amp_mod
        right *= amp_mod

        trem_freq = 0.15 + 0.35 * amp_norm
        trem = 1.0 + 0.3 * amp_norm * np.sin(2 * np.pi * trem_freq * t)
        left *= trem
        right *= trem

        delay_sec = 0.08 + 0.10 * amp_norm
        delay_samples = int(sample_rate * delay_sec)
        decay = 0.45 + 0.25 * amp_norm
        if 0 < delay_samples < n:
            for i in range(delay_samples, n):
                left[i] += left[i - delay_samples] * decay * amp_norm
                right[i] += right[i - delay_samples] * decay * amp_norm

    # ========================
    # 새
    # ========================
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
            if seg_len <= 4:
                continue
            env = np.hanning(seg_len)
            base_freq = np.random.uniform(2200, 4200)
            vibrato = np.sin(
                2 * np.pi * np.linspace(0, 1, seg_len) * np.random.uniform(6, 11)
            )
            phase = np.cumsum(
                2 * np.pi * (base_freq + base_freq * 0.08 * vibrato) / sample_rate
            )
            chirp = np.sin(phase) * env * 0.5
            pan = np.random.uniform(-0.8, 0.8)
            left[start:end] += chirp * (1 - pan)
            right[start:end] += chirp * (1 + pan)

    # ========================
    # 멜로디 (far bell 느낌)
    # ========================
    melody_norm = np.clip(float(melody_intensity) / 100.0, 0.0, 1.0)
    if melody_norm > 0:
        events_per_sec = 0.08 + melody_norm * 0.8
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            length_sec = np.random.uniform(0.6, 1.6)
            start = int(start_time * sample_rate)
            end = min(n, int((start_time + length_sec) * sample_rate))
            seg_len = end - start
            if seg_len <= 4:
                continue
            env = np.linspace(0, 1, seg_len)[::-1] ** 2.5
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

    # ========================
    # 미스테리 (기존)
    # ========================
    pulse_norm = np.clip(float(pulse_intensity) / 100.0, 0.0, 1.0)
    if pulse_norm > 0:
        events_per_sec = 0.08 + pulse_norm * 0.5
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            start = int(start_time * sample_rate)
            length = int(sample_rate * np.random.uniform(0.3, 0.8))
            end = min(n, start + length)
            seg_len = end - start
            if seg_len <= 8:
                continue
            env = np.linspace(0, 1, seg_len)
            env = np.minimum(env, env[::-1]) ** 1.5
            band = np.random.uniform(200, 600)
            rough = np.random.normal(0, 1, seg_len)
            phase = 2 * np.pi * band * np.arange(seg_len) / sample_rate
            rustle = (0.6 * np.sin(phase) + 0.4 * rough) * env * 0.5
            pan = np.random.uniform(-0.6, 0.6)
            left[start:end] += rustle * (1 - pan)
            right[start:end] += rustle * (1 + pan)

    # ========================
    # 기괴함 (악령/크리쳐 느낌)
    # ========================
    weird_norm = np.clip(float(weird_intensity) / 100.0, 0.0, 1.0)
    if weird_norm > 0:
        events_per_sec = 0.03 + weird_norm * 0.3
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            length_sec = np.random.uniform(0.5, 1.5)
            start = int(start_time * sample_rate)
            end = min(n, int((start_time + length_sec) * sample_rate))
            seg_len = end - start
            if seg_len <= 10:
                continue
            env = np.linspace(0, 1, seg_len)
            env = np.minimum(env, env[::-1]) ** 1.2

            base_freq = np.random.uniform(80, 260)
            formant = np.random.uniform(400, 900)
            vibrato = np.sin(
                2 * np.pi * np.linspace(0, 1, seg_len) * np.random.uniform(2, 5)
            )
            phase_low = np.cumsum(
                2 * np.pi * (base_freq + base_freq * 0.2 * vibrato) / sample_rate
            )
            phase_hi = np.cumsum(
                2 * np.pi * (formant + formant * 0.1 * vibrato) / sample_rate
            )
            low = np.sin(phase_low)
            hi = np.sin(phase_hi)
            mix = low * 0.7 + hi * 0.3
            # 살짝 찌그러진 느낌
            creature = np.tanh(mix * 2.0) * env * (0.20 + 0.25 * weird_norm)
            pan = np.random.uniform(-0.7, 0.7)
            left[start:end] += creature * (1 - pan)
            right[start:end] += creature * (1 + pan)

    # ========================
    # 파도
    # ========================
    wave_norm = np.clip(float(wave_intensity) / 100.0, 0.0, 1.0)
    if wave_norm > 0:
        events_per_sec = 0.02 + wave_norm * 0.20
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            length_sec = np.random.uniform(1.5, 4.0)
            start = int(start_time * sample_rate)
            end = min(n, int((start_time + length_sec) * sample_rate))
            seg_len = end - start
            if seg_len <= 10:
                continue
            env = np.linspace(0, 1, seg_len)
            env = np.minimum(env, env[::-1]) ** 1.3
            noise_seg = np.random.normal(0, 1, seg_len)
            # 저역 위주로 살짝 부드럽게
            kernel = np.ones(400, dtype=np.float32) / 400.0
            swell = np.convolve(noise_seg, kernel, mode="same")
            swell = swell / (np.max(np.abs(swell)) + 1e-9)
            swell *= env * (0.20 + 0.30 * wave_norm)
            pan = np.random.uniform(-0.4, 0.4)
            left[start:end] += swell * (1 - pan)
            right[start:end] += swell * (1 + pan)

    # ========================
    # 바람 (멀리서 휘잉)
    # ========================
    wind2_norm = np.clip(float(wind2_intensity) / 100.0, 0.0, 1.0)
    if wind2_norm > 0:
        events_per_sec = 0.03 + wind2_norm * 0.25
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            length_sec = np.random.uniform(1.0, 3.0)
            start = int(start_time * sample_rate)
            end = min(n, int((start_time + length_sec) * sample_rate))
            seg_len = end - start
            if seg_len <= 10:
                continue
            env = np.linspace(0, 1, seg_len)
            env = np.minimum(env, env[::-1]) ** 1.4

            noise_seg = np.random.normal(0, 1, seg_len)
            # 중고역만 남게 약간 하이패스 느낌
            kernel = np.ones(120, dtype=np.float32) / 120.0
            blur = np.convolve(noise_seg, kernel, mode="same")
            wind_noise = noise_seg - 0.6 * blur
            wind_noise = wind_noise / (np.max(np.abs(wind_noise)) + 1e-9)

            # 아주 느린 피치 변동
            lfo = np.sin(
                2 * np.pi * np.linspace(0, 1, seg_len) * np.random.uniform(0.3, 0.7)
            )
            wind_noise *= (1.0 + 0.3 * lfo)

            wind_chunk = wind_noise * env * (0.18 + 0.27 * wind2_norm)
            pan = np.random.uniform(-0.7, 0.7)
            left[start:end] += wind_chunk * (1 - pan)
            right[start:end] += wind_chunk * (1 + pan)

    # ========================
    # 방해 (노이즈 변화)
    # ========================
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
            if seg_len <= 4:
                continue
            fade_len = min(int(sample_rate * 0.02), seg_len // 2)
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            left[start:start + fade_len] *= fade
            right[start:start + fade_len] *= fade
            left[end - fade_len:end] *= fade[::-1]
            right[end - fade_len:end] *= fade[::-1]
            if end - start - 2 * fade_len > 0:
                left[start + fade_len:end - fade_len] *= 0.1
                right[start + fade_len:end - fade_len] *= 0.1

    # ========================
    # 빗자루 (예전 청소)
    # ========================
    snow_norm = np.clip(float(snow_intensity) / 100.0, 0.0, 1.0)
    if snow_norm > 0:
        events_per_sec = 0.20 + snow_norm * 0.8
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            length = int(sample_rate * np.random.uniform(0.05, 0.25))
            start = int(start_time * sample_rate)
            end = min(n, start + length)
            seg_len = end - start
            if seg_len <= 4:
                continue
            env = np.linspace(0, 1, seg_len)
            env = np.minimum(env, env[::-1]) ** 0.8
            noise_seg = np.random.normal(0, 1, seg_len)
            blur = np.convolve(
                noise_seg, np.ones(20, dtype=np.float32) / 20.0, mode="same"
            )
            snow_chunk = (noise_seg - blur) * env * (0.4 + 0.6 * snow_norm)
            pan = np.random.uniform(-0.4, 0.4)
            left[start:end] += snow_chunk * (1 - pan)
            right[start:end] += snow_chunk * (1 + pan)

    # ========================
    # 일렉트로닉 (노이즈 변화, 전자음)
    # ========================
    electro_norm = np.clip(float(electro_intensity) / 100.0, 0.0, 1.0)
    if electro_norm > 0:
        # 전반적인 링모듈레이션 + 약간의 비트크러시 느낌
        carrier_freq = np.random.uniform(8.0, 25.0)
        mod = np.sign(np.sin(2 * np.pi * carrier_freq * t))  # -1 ~ 1
        depth = 0.15 + 0.55 * electro_norm
        left = left * (1.0 - depth) + left * mod * depth
        right = right * (1.0 - depth) + right * mod * depth

        # 간헐적인 글리치 이벤트
        events_per_sec = 0.04 + electro_norm * 0.25
        num_events = int(events_per_sec * duration_sec)
        for _ in range(num_events):
            start_time = np.random.uniform(0, duration_sec)
            length_sec = np.random.uniform(0.08, 0.25)
            start = int(start_time * sample_rate)
            end = min(n, int((start_time + length_sec) * sample_rate))
            seg_len = end - start
            if seg_len <= 4:
                continue
            env = np.linspace(1.0, 0.0, seg_len) ** 1.2
            glitch = np.random.choice([-1.0, 1.0], size=seg_len)
            glitch *= env * (0.18 + 0.25 * electro_norm)
            pan = np.random.uniform(-0.3, 0.3)
            left[start:end] += glitch * (1 - pan)
            right[start:end] += glitch * (1 + pan)

    # leaf_intensity 는 현재 사용 안 함

    # 기본 방향 조정
    dir_norm = np.clip((float(calm) - 50.0) / 50.0, -1.0, 1.0)
    if abs(dir_norm) > 0.01:
        if dir_norm > 0:
            left *= 1.0 - dir_norm
        else:
            right *= 1.0 + dir_norm

    stereo = np.vstack([left, right]).T
    noise_only = stereo.copy()

    # OST 믹스
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

    # 정규화 및 마스터 볼륨
    maxv = np.max(np.abs(stereo)) or 1.0
    stereo = stereo / maxv
    volume_norm = np.clip(float(volume) / 100.0, 0.0, 1.0)
    stereo *= volume_norm

    crossfade_sec = 0.0 if ost_norm > 0 else 1.0
    stereo = make_loopable(stereo, crossfade_sec=crossfade_sec, sample_rate=sample_rate)
    stereo_int16 = (stereo * 32767).astype(np.int16)
    buf = io.BytesIO()
    write(buf, sample_rate, stereo_int16)
    buf.seek(0)
    filename = f"nullwood_{profile}_{duration_sec}s.wav"
    return buf, filename


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
    .eq-bar { width:100%; height:64px; background:rgba(15,23,42,0.7); border-radius:12px; overflow:hidden; margin-bottom:8px; }
    #eqCanvas { width:100%; height:100%; display:block; }
    .noise-toggle-row { margin-bottom:8px; }
    .noise-btn {
        width:100%; border:0;
        padding:14px 10px;
        border-radius:999px;
        font-size:12px;
        cursor:pointer;
        background:#2563eb;
        color:#f9fafb;
        text-align:center;
        line-height:1.2;
    }
    .noise-btn.active { background:#dc2626; color:#fee2e2; }
    .noise-main-label { font-size:13px; font-weight:600; }
    .noise-time-label { font-size:10px; margin-top:2px; opacity:0.9; }
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
    .download-btn {
        flex:1;
        border:0;
        border-radius:10px;
        font-size:12px;
        padding:12px 10px;
        background:#2563eb;
        color:#f9fafb;
        cursor:pointer;
        white-space:nowrap;
    }
    .download-btn:hover { background:#1d4ed8; }
    .download-desc { font-size:11px; color:#9ca3af; margin-top:4px; }
    .ost-links { margin-top:6px; font-size:11px; color:#e5e7eb; line-height:1.4; }
    .ost-links a { color:#93c5fd; text-decoration:none; margin-right:10px; }
    .ost-links a:hover { text-decoration:underline; }
    .footer { margin-top:12px; text-align:center; }
    .footer-logo img { height:48px; opacity:0.9; margin-top:6px; margin-bottom:6px; }
    .footer-links { margin-top:2px; font-size:11px; color:#9ca3af; line-height:1.5; }
    .footer-links a { color:#bfdbfe; text-decoration:none; margin:0 4px; }
    .footer-links a:hover { text-decoration:underline; }
    .footer-license { margin-top:4px; font-size:10px; color:#6b7280; line-height:1.4; }
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
        <button class="noise-btn" id="noiseToggleBtn">
            <div id="noiseMainLabel" class="noise-main-label">Noise On</div>
            <div id="noiseTimeLabel" class="noise-time-label"></div>
        </button>
    </div>

    <label for="volumeRange" data-i18n="volumeLabel" style="margin-top:4px;">마스터 볼륨</label>
    <div class="slider-row">
        <input type="range" id="volumeRange" min="0" max="100" value="56">
        <span class="value" id="volumeValue">56</span>
    </div>

    <label for="calmRange" data-i18n="calmLabel">노이즈 방향 (좌-우)</label>
    <div class="slider-row">
        <input type="range" id="calmRange" min="0" max="100" value="50">
        <span class="value" id="calmValue">50</span>
    </div>

    <label for="lengthRange" data-i18n="lengthLabel">노이즈 길이 (분)</label>
    <div class="slider-row">
        <input type="range" id="lengthRange" min="1" max="60" value="1">
        <span class="value" id="lengthValue">1</span>
    </div>

    <div class="section-title" data-i18n="addTitle">추가 요소</div>

    <label for="timeRange" data-i18n="timeLabel">발자국</label>
    <div class="slider-row">
        <input type="range" id="timeRange" min="0" max="100" value="0">
        <span class="value" id="timeValue">0</span>
    </div>

    <label for="anxietyRange" data-i18n="anxietyLabel">새</label>
    <div class="slider-row">
        <input type="range" id="anxietyRange" min="0" max="100" value="0">
        <span class="value" id="anxietyValue">0</span>
    </div>

    <label for="snowRange" data-i18n="particleLabel">빗자루</label>
    <div class="slider-row">
        <input type="range" id="snowRange" min="0" max="100" value="0">
        <span class="value" id="snowValue">0</span>
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

    <label for="weirdRange" data-i18n="weirdLabel">기괴함</label>
    <div class="slider-row">
        <input type="range" id="weirdRange" min="0" max="100" value="0">
        <span class="value" id="weirdValue">0</span>
    </div>

    <label for="waveRange" data-i18n="waveLabel">파도</label>
    <div class="slider-row">
        <input type="range" id="waveRange" min="0" max="100" value="0">
        <span class="value" id="waveValue">0</span>
    </div>

    <label for="wind2Range" data-i18n="wind2Label">바람</label>
    <div class="slider-row">
        <input type="range" id="wind2Range" min="0" max="100" value="0">
        <span class="value" id="wind2Value">0</span>
    </div>

    <div class="section-title" data-i18n="modTitle">노이즈 변화</div>

    <label for="windRange" data-i18n="windLabel">울림</label>
    <div class="slider-row">
        <input type="range" id="windRange" min="0" max="100" value="0">
        <span class="value" id="windValue">0</span>
    </div>

    <label for="voidRange" data-i18n="voidLabel">방해</label>
    <div class="slider-row">
        <input type="range" id="voidRange" min="0" max="100" value="0">
        <span class="value" id="voidValue">0</span>
    </div>

    <label for="electroRange" data-i18n="electroLabel">일렉트로닉</label>
    <div class="slider-row">
        <input type="range" id="electroRange" min="0" max="100" value="0">
        <span class="value" id="electroValue">0</span>
    </div>

    <label for="ostRange" data-i18n="ostLabel" style="margin-top:10px;">Ephemeral OST 추가</label>
    <div class="slider-row">
        <input type="range" id="ostRange" min="0" max="100" value="0">
        <span class="value" id="ostValue">0</span>
    </div>

    <div class="ost-links">
        <span data-i18n="ostLinksLabel">
            Ephemeral 는 SHHAN의 Nullwood OST의 트랙입니다. 아래 링크를 통해 전체 곡을 들어보세요.
        </span><br>
        <a href="https://youtu.be/BKf4ZT_SiHM?si=dB4XZZfARuK4lcqX" target="_blank">YouTube</a>
        <a href="https://open.spotify.com/track/???some_id" target="_blank">Spotify</a>
    </div>

    <div class="section-title" style="margin-top:14px;" data-i18n="downloadTitle">다운로드</div>
    <div class="download-desc" data-i18n="downloadDesc">
        현재 옵션으로 널우드 노이즈를 다운로드합니다.
    </div>
    <div class="download-row">
        <button class="download-btn" id="downloadBtn" data-i18n="downloadBtn">
            다운로드
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
            as long as you credit: "Audio by SHHAN - Nullwood Noise Generator".
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
        volumeLabel: "Master volume",
        calmLabel: "Noise direction (left-right)",
        lengthLabel: "Noise length (minutes)",
        coreTitle: "Core options",
        addTitle: "Additional sounds",
        modTitle: "Noise shaping",
        timeLabel: "Footsteps",
        windLabel: "Resonance",
        anxietyLabel: "Birds",
        bellLabel: "Melody",
        particleLabel: "Broom",
        pulseLabel: "Mystery",
        weirdLabel: "Creepy voices",
        waveLabel: "Waves",
        wind2Label: "Wind",
        voidLabel: "Interference",
        electroLabel: "Electronic",
        ostLabel: "Add Ephemeral OST",
        ostLinksLabel: "'Ephemeral' is a track from SHHAN's Nullwood OST. Listen to the full piece through the links below.",
        downloadTitle: "Download",
        downloadDesc: "Download Nullwood noise with the current settings.",
        downloadBtn: "Download",
        noiseOn: "Noise On",
        noiseOff: "Noise Off"
    },
    ko: {
        title: "Nullwood 노이즈 생성기",
        subtitle: "Nullwood 노이즈 재생기는 SHHAN의 Nullwood 세계관을 기반으로 한 백색소음 생성기입니다.",
        volumeLabel: "마스터 볼륨",
        calmLabel: "노이즈 방향 (좌-우)",
        lengthLabel: "노이즈 길이 (분)",
        coreTitle: "기본 옵션",
        addTitle: "추가 요소",
        modTitle: "노이즈 변화",
        timeLabel: "발자국",
        windLabel: "울림",
        anxietyLabel: "새",
        bellLabel: "멜로디",
        particleLabel: "빗자루",
        pulseLabel: "미스테리",
        weirdLabel: "기괴함",
        waveLabel: "파도",
        wind2Label: "바람",
        voidLabel: "방해",
        electroLabel: "일렉트로닉",
        ostLabel: "Ephemeral OST 추가",
        ostLinksLabel: "Ephemeral 는 SHHAN의 Nullwood OST의 트랙입니다. 아래 링크를 통해 전체 곡을 들어보세요.",
        downloadTitle: "다운로드",
        downloadDesc: "현재 옵션으로 널우드 노이즈를 다운로드합니다.",
        downloadBtn: "다운로드",
        noiseOn: "노이즈 On",
        noiseOff: "노이즈 Off"
    },
    ja: {
        title: "Nullwood ノイズジェネレーター",
        subtitle: "Nullwoodノイズ再生機はSHHANのNullwood世界観に基づくホワイトノイズ生成機です。",
        volumeLabel: "マスターボリューム",
        calmLabel: "ノイズ方向 (左-右)",
        lengthLabel: "ノイズ長さ (分)",
        coreTitle: "基本オプション",
        addTitle: "追加要素",
        modTitle: "ノイズ変化",
        timeLabel: "足音",
        windLabel: "残響",
        anxietyLabel: "鳥",
        bellLabel: "メロディ",
        particleLabel: "ほうき",
        pulseLabel: "ミステリー",
        weirdLabel: "不気味な声",
        waveLabel: "波の音",
        wind2Label: "風",
        voidLabel: "妨害",
        electroLabel: "エレクトロニック",
        ostLabel: "Ephemeral OST を追加",
        ostLinksLabel: "「Ephemeral」はSHHANのNullwood OSTの1曲です。下のリンクからフルバージョンをお聴きください。",
        downloadTitle: "ダウンロード",
        downloadDesc: "現在の設定でNullwoodノイズをダウンロードします。",
        downloadBtn: "ダウンロード",
        noiseOn: "ノイズ On",
        noiseOff: "ノイズ Off"
    },
    zh: {
        title: "Nullwood 噪音產生器",
        subtitle: "Nullwood噪音播放器是以SHHAN的Nullwood世界觀為基礎的白噪音產生器。",
        volumeLabel: "主音量",
        calmLabel: "噪音方向 (左-右)",
        lengthLabel: "噪音長度 (分鐘)",
        coreTitle: "基本選項",
        addTitle: "附加聲音",
        modTitle: "噪音變化",
        timeLabel: "腳步聲",
        windLabel: "迴響",
        anxietyLabel: "鳥叫",
        bellLabel: "旋律",
        particleLabel: "掃帚",
        pulseLabel: "神祕感",
        weirdLabel: "詭異聲",
        waveLabel: "海浪",
        wind2Label: "風聲",
        voidLabel: "干擾",
        electroLabel: "電子音",
        ostLabel: "加入 Ephemeral OST",
        ostLinksLabel: "「Ephemeral」是SHHAN的Nullwood OST中的一首曲目。請透過下方連結收聽完整樂曲。",
        downloadTitle: "下載",
        downloadDesc: "依照目前設定下載 Nullwood 噪音。",
        downloadBtn: "下載",
        noiseOn: "噪音 On",
        noiseOff: "噪音 Off"
    }
};

let currentLang = "{{ lang }}";
let isNoiseOn = false;
let currentParams = null;

let audio = null;
// 실시간 스트림은 항상 60초 루프
let noiseDurationSec = 60;
let playbackSeconds = 0;

function formatTime(sec) {
    sec = Math.max(0, Math.floor(sec));
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return m.toString() + ":" + (s < 10 ? "0" + s : s.toString());
}

function updateNoiseButtonText() {
    const dict = i18n[currentLang] || i18n["ko"];
    const main = document.getElementById("noiseMainLabel");
    const timeLabel = document.getElementById("noiseTimeLabel");
    if (!main || !timeLabel) return;
    if (isNoiseOn) {
        main.textContent = dict.noiseOff || "Noise Off";
        timeLabel.textContent = formatTime(playbackSeconds) + " / " + formatTime(noiseDurationSec);
    } else {
        main.textContent = dict.noiseOn || "Noise On";
        timeLabel.textContent = "";
    }
}

function applyLanguage(lang) {
    currentLang = lang;
    const dict = i18n[lang] || i18n["ko"];
    document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.getAttribute("data-i18n");
        if (dict[key]) el.textContent = dict[key];
    });
    updateNoiseButtonText();
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

    const drops = [];
    function makeDrop() {
        const w = canvas.width;
        const h = canvas.height;
        return {
            x: Math.random() * w,
            y: Math.random() * h,
            len: 0.15 + Math.random() * 0.25,
            speed: 1.5 + Math.random() * 2.0
        };
    }
    for (let i = 0; i < 80; i++) {
        drops.push(makeDrop());
    }

    function draw() {
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        let vol = 0;
        let dir = 0.5;
        if (currentParams) {
            vol = Number(currentParams.volume || 0) / 100;
            dir = Number(currentParams.calm || 50) / 100;
        }

        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, "rgba(15,23,42,0.9)");
        grad.addColorStop(1, "rgba(3,7,18,0.9)");
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);

        const dropCount = Math.floor(drops.length * (0.2 + vol * 0.8));
        const baseSpeed = 1 + vol * 2.5;

        ctx.lineWidth = 1 * window.devicePixelRatio;
        ctx.strokeStyle = "rgba(255,255,255," + (0.2 + vol * 0.5) + ")";

        const dirBias = (dir - 0.5) * 0.6;

        for (let i = 0; i < dropCount; i++) {
            const d = drops[i];
            const lenPix = d.len * h * (0.6 + vol * 0.8);

            ctx.beginPath();
            ctx.moveTo(d.x, d.y);
            ctx.lineTo(d.x, d.y + lenPix);
            ctx.stroke();

            d.y += d.speed * baseSpeed;
            if (d.y > h + 20) {
                d.y = -Math.random() * h * 0.3;
                let x = Math.random() * w;
                x += dirBias * w;
                if (x < 0) x = Math.random() * w * 0.5;
                if (x > w) x = w * 0.5 + Math.random() * w * 0.5;
                d.x = x;
            }
        }

        requestAnimationFrame(draw);
    }
    draw();
}

function getParams() {
    return {
        profile: "Rain",
        time: document.getElementById("timeRange").value,
        wind: document.getElementById("windRange").value,
        anxiety: document.getElementById("anxietyRange").value,
        calm: document.getElementById("calmRange").value,
        bell: document.getElementById("bellRange").value,
        pulse: document.getElementById("pulseRange").value,
        weird: document.getElementById("weirdRange").value,
        wave: document.getElementById("waveRange").value,
        wind2: document.getElementById("wind2Range").value,
        void: document.getElementById("voidRange").value,
        particle: document.getElementById("snowRange").value,
        electro: document.getElementById("electroRange").value,
        ost: document.getElementById("ostRange").value,
        volume: document.getElementById("volumeRange").value,
        length: document.getElementById("lengthRange").value
    };
}

let updateTimer = null;

function updateAudio() {
    if (!isNoiseOn || !audio) return;
    const p = getParams();
    currentParams = p;

    // 실시간 스트림은 항상 1분짜리 루프 (생성 시간 문제 방지)
    noiseDurationSec = 60;

    const url = new URL("/stream", window.location.origin);
    Object.entries(p).forEach(([k, v]) => url.searchParams.set(k, v));
    url.searchParams.set("minutes", "1");

    audio.src = url.toString();
    audio.loop = true;
    playbackSeconds = 0;
    updateNoiseButtonText();
    audio.play().catch(() => {});
}

function scheduleUpdate() {
    if (!isNoiseOn) return;
    if (updateTimer) clearTimeout(updateTimer);
    updateTimer = setTimeout(updateAudio, 250);
}

function initAudio() {
    audio = new Audio();
    audio.loop = true;
    audio.addEventListener("timeupdate", () => {
        if (!isNoiseOn) return;
        playbackSeconds = Math.floor(audio.currentTime);
        if (playbackSeconds > noiseDurationSec) {
            playbackSeconds = playbackSeconds % noiseDurationSec;
        }
        updateNoiseButtonText();
    });
}

function setupControls() {
    const ids = [
        "volumeRange","calmRange","lengthRange",
        "timeRange","anxietyRange","snowRange",
        "bellRange","pulseRange",
        "weirdRange","waveRange","wind2Range",
        "windRange","voidRange","electroRange",
        "ostRange"
    ];
    ids.forEach(id => {
        const el = document.getElementById(id);
        const valSpan = document.getElementById(id.replace("Range","Value"));
        if (el && el.type === "range") {
            el.addEventListener("input", () => {
                if (valSpan) valSpan.textContent = el.value;
                if (id === "lengthRange") {
                    // 길이는 다운로드 전용이지만 값 표시를 위해 업데이트만
                    return;
                }
                scheduleUpdate();
            });
        }
    });

    document.getElementById("volumeRange").addEventListener("input", e => {
        document.getElementById("volumeValue").textContent = e.target.value;
    });

    const noiseBtn = document.getElementById("noiseToggleBtn");
    noiseBtn.addEventListener("click", () => {
        isNoiseOn = !isNoiseOn;
        noiseBtn.classList.toggle("active", isNoiseOn);
        if (isNoiseOn) {
            updateAudio();
        } else if (audio) {
            audio.pause();
            audio.currentTime = 0;
            playbackSeconds = 0;
            updateNoiseButtonText();
        }
    });

    document.getElementById("downloadBtn").addEventListener("click", () => {
        const p = getParams();
        const minutes = Math.min(60, Math.max(1, parseInt(p.length || "1", 10)));
        const url = new URL("/download", window.location.origin);
        Object.entries(p).forEach(([k, v]) => url.searchParams.set(k, v));
        url.searchParams.set("minutes", minutes.toString());
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
    initAudio();
    setupEQ();
    applyLanguage("ko");
    setupControls();
    currentParams = getParams();
    updateNoiseButtonText();
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
    weird = int(request.args.get("weird", 0))
    wave = int(request.args.get("wave", 0))
    wind2 = int(request.args.get("wind2", 0))
    void = int(request.args.get("void", 0))
    snow = int(request.args.get("particle", 0))
    electro = int(request.args.get("electro", 0))
    leaf = int(request.args.get("leaf", 0))
    ost = int(request.args.get("ost", 0))
    volume = int(request.args.get("volume", 56))

    # 실시간 스트림은 항상 1분 길이
    duration_sec = 60

    buf, filename = generate_nullwood_noise(
        duration_sec=duration_sec,
        profile=profile,
        time_intensity=time_int,
        amp_intensity=amp,
        anxiety=anxiety,
        calm=calm,
        melody_intensity=melody,
        pulse_intensity=pulse,
        weird_intensity=weird,
        wave_intensity=wave,
        wind2_intensity=wind2,
        void_intensity=void,
        snow_intensity=snow,
        electro_intensity=electro,
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
    weird = int(request.args.get("weird", 0))
    wave = int(request.args.get("wave", 0))
    wind2 = int(request.args.get("wind2", 0))
    void = int(request.args.get("void", 0))
    snow = int(request.args.get("particle", 0))
    electro = int(request.args.get("electro", 0))
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
        weird_intensity=weird,
        wave_intensity=wave,
        wind2_intensity=wind2,
        void_intensity=void,
        snow_intensity=snow,
        electro_intensity=electro,
        leaf_intensity=leaf,
        ost_mix=ost,
        volume=volume,
    )
    download_name = f"{filename[:-4]}_{minutes}min.wav"
    return send_file(buf, as_attachment=True, download_name=download_name, mimetype="audio/wav")


if __name__ == "__main__":
    load_ost()
    app.run(host="127.0.0.1", port=5000, debug=True)
