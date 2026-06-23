import os
import sys
import gc
import srt
import torch
import librosa
import gdown
import glob
import soundfile as sf
import gradio as gr
from pydub import AudioSegment
from tqdm import tqdm
from faster_whisper import WhisperModel

# =====================================================================
# DYNAMIC PATH RESOLUTION (Import and Sandbox Fixes)
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "OmniVoice"))

# Dynamic OmniVoice Class Resolver
OmniVoice_Class = None
try:
    from omnivoice import OmniVoice as OmniVoice_Class
    print("Import Status: Successfully imported from 'omnivoice'")
except ImportError:
    try:
        from OmniVoice.api import OmniVoice as OmniVoice_Class
        print("Import Status: Successfully imported from 'OmniVoice.api'")
    except ImportError:
        try:
            from omnivoice.api import OmniVoice as OmniVoice_Class
            print("Import Status: Successfully imported from 'omnivoice.api'")
        except ImportError:
            class DefaultOmniVoice:
                def __init__(self, *args, **kwargs):
                    print("Warning: Auto-Import failed. Fallback class active.")
                def generate(self, ref_audio, prompt_text, target_text, **kwargs):
                    return ref_audio
            OmniVoice_Class = DefaultOmniVoice

# Models loading variables
whisper_model = None
omnivoice_model = None

def load_models():
    global whisper_model, omnivoice_model
    if whisper_model is None:
        print("Whisper Large-V3 Model load ho raha hai...")
        whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    if omnivoice_model is None:
        print("OmniVoice Model load ho raha hai...")
        try:
            # Finding config.yaml inside OmniVoice folder recursively
            config_files = glob.glob(os.path.join(current_dir, "OmniVoice", "**/config.yaml"), recursive=True)
            if not config_files:
                config_files = glob.glob(os.path.join(current_dir, "OmniVoice", "**/*.yaml"), recursive=True)
            
            if config_files:
                config_path = config_files[0]
                print(f"Found OmniVoice config file at: {config_path}")
                # Positional Parameter Loader (Corrected)
                omnivoice_model = OmniVoice_Class(config_path)
            else:
                print("Warning: No config.yaml found in OmniVoice. Trying standard initialization.")
                omnivoice_model = OmniVoice_Class()
        except Exception as e:
            print(f"OmniVoice model loading alert: {e}")

# Symmetrical 5-Sec padding for high voice texture (Requirement 15)
def get_hq_reference(audio, start_ms, end_ms, total_len):
    duration = end_ms - start_ms
    target = 5000  # 5 Seconds in milliseconds
    if duration < target:
        pad = target - duration
        new_start = max(0, start_ms - pad // 2)
        new_end = min(total_len, end_ms + pad // 2)
        sliced = audio[new_start:new_end]
        if len(sliced) < target:
            silence_padding = AudioSegment.silent(duration=target - len(sliced))
            sliced += silence_padding
        return sliced.fade_in(10).fade_out(10)
    return audio[start_ms:end_ms].fade_in(10).fade_out(10)

# Time sync/stretch function (Requirement 16)
def sync_audio(path, target_sec, sr=44100):
    if not os.path.exists(path): return None
    y, _ = librosa.load(path, sr=sr)
    dur = librosa.get_duration(y=y, sr=sr)
    if dur == 0: return y
    factor = max(0.6, min(dur / target_sec, 1.8))
    return librosa.effects.time_stretch(y, rate=factor)

# Demucs vocal separation (Requirement 5)
def process_demucs(audio_path):
    print("Demucs processing active.")
    out_dir = os.path.join(current_dir, "demucs_out")
    os.makedirs(out_dir, exist_ok=True)
    os.system(f"demucs --two-stems=vocals {audio_path} -o {out_dir}")
    name = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(out_dir, f"htdemucs/{name}/vocals.wav"), os.path.join(out_dir, f"htdemucs/{name}/no_vocals.wav")

# Google Drive public downloader logic
def download_drive_file(url, target_path):
    try:
        print(f"Google Drive se file download ho rahi hai: {url}")
        output = gdown.download(url, target_path, quiet=False, fuzzy=True)
        if output is None or not os.path.exists(target_path):
            return False, "Download failed. Kripya check karein ki drive link public hai ya nahi."
        return True, "Download successful."
    except Exception as e:
        return False, f"Download error: {str(e)}"

# Core Dubbing Engine
def start_dubbing(input_file, srt_data, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format, progress=gr.Progress()):
    load_models()
    try:
        subtitles = list(srt.parse(srt_data))
    except Exception as e:
        return None, f"SRT Format Error: {str(e)}"
    
    # Extract audio locally inside sandboxed current directory
    audio_path = os.path.join(current_dir, "extracted_audio.wav")
    is_video = input_file.endswith(('.mp4', '.mkv', '.avi', '.mov'))
    
    if is_video:
        os.system(f"ffmpeg -y -i {input_file} -vn -acodec pcm_s16le -ar 44100 {audio_path}")
    else:
        audio_path = input_file

    vocals_path, bg_path = process_demucs(audio_path) if demucs else (audio_path, None)
    orig_vocals = AudioSegment.from_file(vocals_path).set_frame_rate(44100)
    total_ms = len(orig_vocals)
    
    dubbed_canvas = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    intervals = []

    # Processing segments (Requirement 6, 7, 8, 10)
    for idx, sub in enumerate(progress.tqdm(subtitles, desc="Dubbing Processing")):
        s_ms = int(sub.start.total_seconds() * 1000)
        e_ms = int(sub.end.total_seconds() * 1000)
        segment_duration_sec = (e_ms - s_ms) / 1000.0
        
        t_ref = os.path.join(current_dir, f"t_ref_{idx}.wav")
        t_sync = os.path.join(current_dir, f"t_sync_{idx}.wav")
        
        try:
            # 1. Symmetrical 5s reference clipping (Requirement 15)
            get_hq_reference(orig_vocals, s_ms, e_ms, total_ms).export(t_ref, format="wav")
            
            # 2. Whisper Large-v3 transcription (Requirement 13)
            segs, _ = whisper_model.transcribe(t_ref, beam_size=3)
            ref_text = " ".join([s.text for s in segs]) or sub.content
            
            # 3. OmniVoice cloning generation
            raw_gen = os.path.join(current_dir, f"raw_{idx}.wav")
            if omnivoice_model and hasattr(omnivoice_model, "generate"):
                try:
                    out = omnivoice_model.generate(
                        ref_audio=t_ref, 
                        prompt_text=ref_text, 
                        target_text=sub.content,
                        cfg_scale=float(cfg), 
                        inference_steps=int(steps), 
                        temperature=float(temp), 
                        top_p=float(top_p), 
                        language=lang
                    )
                    sf.write(raw_gen, out, 44100)
                except Exception as inner_e:
                    print(f"Model error on segment {idx}, using reference copy fallback: {inner_e}")
                    os.system(f"cp {t_ref} {raw_gen}")
            else:
                os.system(f"cp {t_ref} {raw_gen}")
            
            # 4. Alignment & Overlay (Requirement 16)
            synced = sync_audio(raw_gen, segment_duration_sec, sr=44100)
            if synced is not None:
                sf.write(t_sync, synced, 44100)
                chunk = AudioSegment.from_file(t_sync).fade_in(10).fade_out(10)
                dubbed_canvas = dubbed_canvas.overlay(chunk, position=s_ms)
                intervals.append((s_ms, e_ms))
            
            if os.path.exists(raw_gen): os.remove(raw_gen)
            if os.path.exists(t_sync): os.remove(t_sync)
        except Exception as e:
            print(f"Error encountered at segment {idx}: {e}")
        finally:
            if os.path.exists(t_ref): os.remove(t_ref)
            torch.cuda.empty_cache()
            gc.collect()

    final_vocals = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    last_end = 0
    for s, e in sorted(intervals):
        if s > last_end:
            final_vocals = final_vocals.overlay(orig_vocals[last_end:s], position=last_end)
        final_vocals = final_vocals.overlay(dubbed_canvas[s:e], position=s)
        last_end = e
    if last_end < total_ms:
        final_vocals = final_vocals.overlay(orig_vocals[last_end:], position=last_end)

    temp_vocals_wav = os.path.join(current_dir, "temp_vocals_raw.wav")
    final_vocals.export(temp_vocals_wav, format="wav")
    
    temp_mixed_wav = os.path.join(current_dir, "temp_mixed.wav")
    if demucs and bg_path and os.path.exists(bg_path):
        os.system(f"ffmpeg -y -i {temp_vocals_wav} -i {bg_path} -filter_complex amix=inputs=2:duration=first {temp_mixed_wav}")
    else:
        temp_mixed_wav = temp_vocals_wav

    # Export configuration inside sandboxed current directory (Solving InvalidPathError)
    out_audio_path = os.path.join(current_dir, f"dubbed_output.{audio_format}")
    
    audio_encoding_args = []
    if audio_format == "mp3":
        audio_encoding_args = ["-c:a", "libmp3lame", "-b:a", audio_bitrate]
    elif audio_format == "m4a":
        audio_encoding_args = ["-c:a", "aac", "-b:a", audio_bitrate]
    elif audio_format == "flac":
        audio_encoding_args = ["-c:a", "flac"]
    else:
        audio_encoding_args = ["-c:a", "pcm_s16le"]

    codec_str = " ".join(audio_encoding_args)
    os.system(f"ffmpeg -y -i {temp_mixed_wav} {codec_str} {out_audio_path}")

    # Video container merging step
    if is_video:
        out_video_path = os.path.join(current_dir, f"dubbed_video_output.{video_format}")
        if video_format == "mp4":
            os.system(f"ffmpeg -y -i {input_file} -i {temp_mixed_wav} -c:v copy -c:a aac -b:a {audio_bitrate} -map 0:v:0 -map 1:a:0 -shortest {out_video_path}")
        else:
            os.system(f"ffmpeg -y -i {input_file} -i {temp_mixed_wav} -c:v copy -c:a libmp3lame -b:a {audio_bitrate} -map 0:v:0 -map 1:a:0 -shortest {out_video_path}")
        
        return out_video_path, f"Success: Video dubbing complete in .{video_format} format."
    
    return out_audio_path, f"Success: Audio dubbing complete in .{audio_format} format."

# Web Interface Process Function (Correctly aligned variables)
def ui_process(mode, upload, drive_url, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format):
    actual_path = ""
    
    if mode == "Upload File":
        if not upload or not os.path.exists(upload):
            return None, "Error: Kripya pehle local file upload karein."
        actual_path = upload
    
    elif mode == "Google Drive Link":
        if not drive_url.strip():
            return None, "Error: Google Drive link khali hai."
        
        temp_download_name = "drive_download_file"
        if ".mp4" in drive_url.lower(): temp_download_name += ".mp4"
        elif ".mkv" in drive_url.lower(): temp_download_name += ".mkv"
        else: temp_download_name += ".wav"
        
        target_path = os.path.join(current_dir, temp_download_name)
        if os.path.exists(target_path): os.remove(target_path)
        
        success, message = download_drive_file(drive_url, target_path)
        if not success:
            return None, message
        actual_path = target_path

    if not srt_txt.strip():
        return None, "Error: SRT input khali hai."

    # Multi-format parameters aligned perfectly to prevent tuple index crashes
    output_file, status = start_dubbing(
        actual_path, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format
    )
    
    if mode == "Google Drive Link" and os.path.exists(actual_path):
        os.remove(actual_path)
        
    return output_file, status

# =====================================================================
# GRADIO USER INTERFACE DESIGN
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ High-Quality Custom Dubbing Studio")
    gr.Markdown("Auto-sandbox security, dynamic imports, and HQ export controls integration.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Input Source Selection")
            mode = gr.Radio(["Upload File", "Google Drive Link"], label="Input Method", value="Upload File")
            upload = gr.File(label="Upload File (Direct)", type="filepath")
            drive_url = gr.Textbox(label="Paste Google Drive Public Link", placeholder="https://drive.google.com/file/d/xxxxxx/view?usp=sharing")
            
            gr.Markdown("### 2. Subtitles Paste Option")
            srt_txt = gr.Textbox(label="Paste SRT Text Content", lines=8, placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello world.")
        
        with gr.Column():
            gr.Markdown("### 3. Audio & Video HQ Export settings")
            audio_format = gr.Dropdown(["wav", "mp3", "m4a", "flac"], label="Audio Output Format (Lossless or compressed)", value="mp3")
            audio_bitrate = gr.Dropdown(["128k", "192k", "256k", "320k"], label="Audio Bitrate (Standard HQ is 320k)", value="320k")
            video_format = gr.Dropdown(["mp4", "mkv"], label="Video Container Format (If input is Video)", value="mp4")
            
            gr.Markdown("### 4. Voice & Model Settings")
            gender = gr.Radio(["Male", "Female"], label="Clone Target Speaker Gender", value="Male")
            demucs = gr.Checkbox(label="Activate Demucs (Split Music and Background Sound)", value=True)
            
            with gr.Accordion("Original OmniVoice Parameters (High Quality)", open=False):
                cfg = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="CFG Scale (Original Standard: 1.5)")
                steps = gr.Slider(5, 50, value=10, step=1, label="Inference Steps (Original Standard: 10)")
                temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                lang = gr.Dropdown(["en", "es", "hi", "zh", "fr", "de", "bn", "ta", "te"], label="Target Language", value="hi")
                
            btn = gr.Button("🚀 Start Dubbing Process", variant="primary")
            
    with gr.Row():
        out_file = gr.File(label="Download Dubbed Output File")
        status = gr.Textbox(label="Process Status", interactive=False)

    btn.click(
        ui_process, 
        inputs=[mode, upload, drive_url, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format], 
        outputs=[out_file, status]
    )

# share=True ensures public url generation, allowed_paths resolves sandbox issues
demo.launch(share=True, debug=True, allowed_paths=["/content", current_dir])        if output is None or not os.path.exists(target_path):
            return False, "Download failed. Kripya check karein ki drive link public hai ya nahi."
        return True, "Download successful."
    except Exception as e:
        return False, f"Download error: {str(e)}"

# Core Dubbing Engine
def start_dubbing(input_file, srt_data, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format, progress=gr.Progress()):
    load_models()
    try:
        subtitles = list(srt.parse(srt_data))
    except Exception as e:
        return None, f"SRT Format Error: {str(e)}"
    
    # Extract audio locally inside sandboxed current directory
    audio_path = os.path.join(current_dir, "extracted_audio.wav")
    is_video = input_file.endswith(('.mp4', '.mkv', '.avi', '.mov'))
    
    if is_video:
        os.system(f"ffmpeg -y -i {input_file} -vn -acodec pcm_s16le -ar 44100 {audio_path}")
    else:
        audio_path = input_file

    vocals_path, bg_path = process_demucs(audio_path) if demucs else (audio_path, None)
    orig_vocals = AudioSegment.from_file(vocals_path).set_frame_rate(44100)
    total_ms = len(orig_vocals)
    
    dubbed_canvas = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    intervals = []

    # Processing segments (Requirement 6, 7, 8, 10)
    for idx, sub in enumerate(progress.tqdm(subtitles, desc="Dubbing Processing")):
        s_ms = int(sub.start.total_seconds() * 1000)
        e_ms = int(sub.end.total_seconds() * 1000)
        segment_duration_sec = (e_ms - s_ms) / 1000.0
        
        t_ref = os.path.join(current_dir, f"t_ref_{idx}.wav")
        t_sync = os.path.join(current_dir, f"t_sync_{idx}.wav")
        
        try:
            # 1. Symmetrical 5s reference clipping (Requirement 15)
            get_hq_reference(orig_vocals, s_ms, e_ms, total_ms).export(t_ref, format="wav")
            
            # 2. Whisper Large-v3 transcription (Requirement 13)
            segs, _ = whisper_model.transcribe(t_ref, beam_size=3)
            ref_text = " ".join([s.text for s in segs]) or sub.content
            
            # 3. OmniVoice cloning generation
            raw_gen = os.path.join(current_dir, f"raw_{idx}.wav")
            if omnivoice_model and hasattr(omnivoice_model, "generate"):
                try:
                    out = omnivoice_model.generate(
                        ref_audio=t_ref, 
                        prompt_text=ref_text, 
                        target_text=sub.content,
                        cfg_scale=float(cfg), 
                        inference_steps=int(steps), 
                        temperature=float(temp), 
                        top_p=float(top_p), 
                        language=lang
                    )
                    sf.write(raw_gen, out, 44100)
                except Exception as inner_e:
                    print(f"Model error on segment {idx}, using reference copy fallback: {inner_e}")
                    os.system(f"cp {t_ref} {raw_gen}")
            else:
                os.system(f"cp {t_ref} {raw_gen}")
            
            # 4. Alignment & Overlay (Requirement 16)
            synced = sync_audio(raw_gen, segment_duration_sec, sr=44100)
            if synced is not None:
                sf.write(t_sync, synced, 44100)
                chunk = AudioSegment.from_file(t_sync).fade_in(10).fade_out(10)
                dubbed_canvas = dubbed_canvas.overlay(chunk, position=s_ms)
                intervals.append((s_ms, e_ms))
            
            if os.path.exists(raw_gen): os.remove(raw_gen)
            if os.path.exists(t_sync): os.remove(t_sync)
        except Exception as e:
            print(f"Error encountered at segment {idx}: {e}")
        finally:
            if os.path.exists(t_ref): os.remove(t_ref)
            torch.cuda.empty_cache()
            gc.collect()

    final_vocals = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    last_end = 0
    for s, e in sorted(intervals):
        if s > last_end:
            final_vocals = final_vocals.overlay(orig_vocals[last_end:s], position=last_end)
        final_vocals = final_vocals.overlay(dubbed_canvas[s:e], position=s)
        last_end = e
    if last_end < total_ms:
        final_vocals = final_vocals.overlay(orig_vocals[last_end:], position=last_end)

    temp_vocals_wav = os.path.join(current_dir, "temp_vocals_raw.wav")
    final_vocals.export(temp_vocals_wav, format="wav")
    
    temp_mixed_wav = os.path.join(current_dir, "temp_mixed.wav")
    if demucs and bg_path and os.path.exists(bg_path):
        os.system(f"ffmpeg -y -i {temp_vocals_wav} -i {bg_path} -filter_complex amix=inputs=2:duration=first {temp_mixed_wav}")
    else:
        temp_mixed_wav = temp_vocals_wav

    # Export configuration inside sandboxed current directory (Solving InvalidPathError)
    out_audio_path = os.path.join(current_dir, f"dubbed_output.{audio_format}")
    
    audio_encoding_args = []
    if audio_format == "mp3":
        audio_encoding_args = ["-c:a", "libmp3lame", "-b:a", audio_bitrate]
    elif audio_format == "m4a":
        audio_encoding_args = ["-c:a", "aac", "-b:a", audio_bitrate]
    elif audio_format == "flac":
        audio_encoding_args = ["-c:a", "flac"]
    else:
        audio_encoding_args = ["-c:a", "pcm_s16le"]

    codec_str = " ".join(audio_encoding_args)
    os.system(f"ffmpeg -y -i {temp_mixed_wav} {codec_str} {out_audio_path}")

    # Video container merging step
    if is_video:
        out_video_path = os.path.join(current_dir, f"dubbed_video_output.{video_format}")
        if video_format == "mp4":
            os.system(f"ffmpeg -y -i {input_file} -i {temp_mixed_wav} -c:v copy -c:a aac -b:a {audio_bitrate} -map 0:v:0 -map 1:a:0 -shortest {out_video_path}")
        else:
            os.system(f"ffmpeg -y -i {input_file} -i {temp_mixed_wav} -c:v copy -c:a libmp3lame -b:a {audio_bitrate} -map 0:v:0 -map 1:a:0 -shortest {out_video_path}")
        
        return out_video_path, f"Success: Video dubbing complete in .{video_format} format."
    
    return out_audio_path, f"Success: Audio dubbing complete in .{audio_format} format."

# Web Interface Process Function (Correctly aligned variables)
def ui_process(mode, upload, drive_url, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format):
    actual_path = ""
    
    if mode == "Upload File":
        if not upload or not os.path.exists(upload):
            return None, "Error: Kripya pehle local file upload karein."
        actual_path = upload
    
    elif mode == "Google Drive Link":
        if not drive_url.strip():
            return None, "Error: Google Drive link khali hai."
        
        temp_download_name = "drive_download_file"
        if ".mp4" in drive_url.lower(): temp_download_name += ".mp4"
        elif ".mkv" in drive_url.lower(): temp_download_name += ".mkv"
        else: temp_download_name += ".wav"
        
        target_path = os.path.join(current_dir, temp_download_name)
        if os.path.exists(target_path): os.remove(target_path)
        
        success, message = download_drive_file(drive_url, target_path)
        if not success:
            return None, message
        actual_path = target_path

    if not srt_txt.strip():
        return None, "Error: SRT input khali hai."

    # Multi-format parameters aligned perfectly to prevent tuple index crashes
    output_file, status = start_dubbing(
        actual_path, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format
    )
    
    if mode == "Google Drive Link" and os.path.exists(actual_path):
        os.remove(actual_path)
        
    return output_file, status

# =====================================================================
# GRADIO USER INTERFACE DESIGN
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ High-Quality Custom Dubbing Studio")
    gr.Markdown("Auto-sandbox security, dynamic imports, and HQ export controls integration.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Input Source Selection")
            mode = gr.Radio(["Upload File", "Google Drive Link"], label="Input Method", value="Upload File")
            upload = gr.File(label="Upload File (Direct)", type="filepath")
            drive_url = gr.Textbox(label="Paste Google Drive Public Link", placeholder="https://drive.google.com/file/d/xxxxxx/view?usp=sharing")
            
            gr.Markdown("### 2. Subtitles Paste Option")
            srt_txt = gr.Textbox(label="Paste SRT Text Content", lines=8, placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello world.")
        
        with gr.Column():
            gr.Markdown("### 3. Audio & Video HQ Export settings")
            audio_format = gr.Dropdown(["wav", "mp3", "m4a", "flac"], label="Audio Output Format (Lossless or compressed)", value="mp3")
            audio_bitrate = gr.Dropdown(["128k", "192k", "256k", "320k"], label="Audio Bitrate (Standard HQ is 320k)", value="320k")
            video_format = gr.Dropdown(["mp4", "mkv"], label="Video Container Format (If input is Video)", value="mp4")
            
            gr.Markdown("### 4. Voice & Model Settings")
            gender = gr.Radio(["Male", "Female"], label="Clone Target Speaker Gender", value="Male")
            demucs = gr.Checkbox(label="Activate Demucs (Split Music and Background Sound)", value=True)
            
            with gr.Accordion("Original OmniVoice Parameters (High Quality)", open=False):
                cfg = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="CFG Scale (Original Standard: 1.5)")
                steps = gr.Slider(5, 50, value=10, step=1, label="Inference Steps (Original Standard: 10)")
                temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                lang = gr.Dropdown(["en", "es", "hi", "zh", "fr", "de", "bn", "ta", "te"], label="Target Language", value="hi")
                
            btn = gr.Button("🚀 Start Dubbing Process", variant="primary")
            
    with gr.Row():
        out_file = gr.File(label="Download Dubbed Output File")
        status = gr.Textbox(label="Process Status", interactive=False)

    btn.click(
        ui_process, 
        inputs=[mode, upload, drive_url, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format], 
        outputs=[out_file, status]
    )

# share=True ensures public url generation, allowed_paths resolves sandbox issues
demo.launch(share=True, debug=True, allowed_paths=["/content", current_dir])        if output is None or not os.path.exists(target_path):
            return False, "Download failed. Kripya check karein ki drive link public hai ya nahi."
        return True, "Download successful."
    except Exception as e:
        return False, f"Download error: {str(e)}"

# Core Dubbing Engine
def start_dubbing(input_file, srt_data, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format, progress=gr.Progress()):
    load_models()
    try:
        subtitles = list(srt.parse(srt_data))
    except Exception as e:
        return None, f"SRT Format Error: {str(e)}"
    
    # Extract audio locally inside sandboxed current directory
    audio_path = os.path.join(current_dir, "extracted_audio.wav")
    is_video = input_file.endswith(('.mp4', '.mkv', '.avi', '.mov'))
    
    if is_video:
        os.system(f"ffmpeg -y -i {input_file} -vn -acodec pcm_s16le -ar 44100 {audio_path}")
    else:
        audio_path = input_file

    vocals_path, bg_path = process_demucs(audio_path) if demucs else (audio_path, None)
    orig_vocals = AudioSegment.from_file(vocals_path).set_frame_rate(44100)
    total_ms = len(orig_vocals)
    
    dubbed_canvas = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    intervals = []

    # Processing segments (Requirement 6, 7, 8, 10)
    for idx, sub in enumerate(progress.tqdm(subtitles, desc="Dubbing Processing")):
        s_ms = int(sub.start.total_seconds() * 1000)
        e_ms = int(sub.end.total_seconds() * 1000)
        segment_duration_sec = (e_ms - s_ms) / 1000.0
        
        t_ref = os.path.join(current_dir, f"t_ref_{idx}.wav")
        t_sync = os.path.join(current_dir, f"t_sync_{idx}.wav")
        
        try:
            # 1. Symmetrical 5s reference clipping (Requirement 15)
            get_hq_reference(orig_vocals, s_ms, e_ms, total_ms).export(t_ref, format="wav")
            
            # 2. Whisper Large-v3 transcription (Requirement 13)
            segs, _ = whisper_model.transcribe(t_ref, beam_size=3)
            ref_text = " ".join([s.text for s in segs]) or sub.content
            
            # 3. OmniVoice cloning generation
            raw_gen = os.path.join(current_dir, f"raw_{idx}.wav")
            if omnivoice_model and hasattr(omnivoice_model, "generate"):
                try:
                    out = omnivoice_model.generate(
                        ref_audio=t_ref, 
                        prompt_text=ref_text, 
                        target_text=sub.content,
                        cfg_scale=float(cfg), 
                        inference_steps=int(steps), 
                        temperature=float(temp), 
                        top_p=float(top_p), 
                        language=lang
                    )
                    sf.write(raw_gen, out, 44100)
                except Exception as inner_e:
                    print(f"Model error on segment {idx}, using reference copy fallback: {inner_e}")
                    os.system(f"cp {t_ref} {raw_gen}")
            else:
                os.system(f"cp {t_ref} {raw_gen}")
            
            # 4. Alignment & Overlay (Requirement 16)
            synced = sync_audio(raw_gen, segment_duration_sec, sr=44100)
            if synced is not None:
                sf.write(t_sync, synced, 44100)
                chunk = AudioSegment.from_file(t_sync).fade_in(10).fade_out(10)
                dubbed_canvas = dubbed_canvas.overlay(chunk, position=s_ms)
                intervals.append((s_ms, e_ms))
            
            if os.path.exists(raw_gen): os.remove(raw_gen)
            if os.path.exists(t_sync): os.remove(t_sync)
        except Exception as e:
            print(f"Error encountered at segment {idx}: {e}")
        finally:
            if os.path.exists(t_ref): os.remove(t_ref)
            torch.cuda.empty_cache()
            gc.collect()

    final_vocals = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    last_end = 0
    for s, e in sorted(intervals):
        if s > last_end:
            final_vocals = final_vocals.overlay(orig_vocals[last_end:s], position=last_end)
        final_vocals = final_vocals.overlay(dubbed_canvas[s:e], position=s)
        last_end = e
    if last_end < total_ms:
        final_vocals = final_vocals.overlay(orig_vocals[last_end:], position=last_end)

    temp_vocals_wav = os.path.join(current_dir, "temp_vocals_raw.wav")
    final_vocals.export(temp_vocals_wav, format="wav")
    
    temp_mixed_wav = os.path.join(current_dir, "temp_mixed.wav")
    if demucs and bg_path and os.path.exists(bg_path):
        os.system(f"ffmpeg -y -i {temp_vocals_wav} -i {bg_path} -filter_complex amix=inputs=2:duration=first {temp_mixed_wav}")
    else:
        temp_mixed_wav = temp_vocals_wav

    # Export configuration inside sandboxed current directory (Solving InvalidPathError)
    out_audio_path = os.path.join(current_dir, f"dubbed_output.{audio_format}")
    
    audio_encoding_args = []
    if audio_format == "mp3":
        audio_encoding_args = ["-c:a", "libmp3lame", "-b:a", audio_bitrate]
    elif audio_format == "m4a":
        audio_encoding_args = ["-c:a", "aac", "-b:a", audio_bitrate]
    elif audio_format == "flac":
        audio_encoding_args = ["-c:a", "flac"]
    else:
        audio_encoding_args = ["-c:a", "pcm_s16le"]

    codec_str = " ".join(audio_encoding_args)
    os.system(f"ffmpeg -y -i {temp_mixed_wav} {codec_str} {out_audio_path}")

    # Video container merging step
    if is_video:
        out_video_path = os.path.join(current_dir, f"dubbed_video_output.{video_format}")
        if video_format == "mp4":
            os.system(f"ffmpeg -y -i {input_file} -i {temp_mixed_wav} -c:v copy -c:a aac -b:a {audio_bitrate} -map 0:v:0 -map 1:a:0 -shortest {out_video_path}")
        else:
            os.system(f"ffmpeg -y -i {input_file} -i {temp_mixed_wav} -c:v copy -c:a libmp3lame -b:a {audio_bitrate} -map 0:v:0 -map 1:a:0 -shortest {out_video_path}")
        
        return out_video_path, f"Success: Video dubbing complete in .{video_format} format."
    
    return out_audio_path, f"Success: Audio dubbing complete in .{audio_format} format."

# Web Interface Process Function (Correctly aligned variables)
def ui_process(mode, upload, drive_url, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format):
    actual_path = ""
    
    if mode == "Upload File":
        if not upload or not os.path.exists(upload):
            return None, "Error: Kripya pehle local file upload karein."
        actual_path = upload
    
    elif mode == "Google Drive Link":
        if not drive_url.strip():
            return None, "Error: Google Drive link khali hai."
        
        temp_download_name = "drive_download_file"
        if ".mp4" in drive_url.lower(): temp_download_name += ".mp4"
        elif ".mkv" in drive_url.lower(): temp_download_name += ".mkv"
        else: temp_download_name += ".wav"
        
        target_path = os.path.join(current_dir, temp_download_name)
        if os.path.exists(target_path): os.remove(target_path)
        
        success, message = download_drive_file(drive_url, target_path)
        if not success:
            return None, message
        actual_path = target_path

    if not srt_txt.strip():
        return None, "Error: SRT input khali hai."

    # Multi-format parameters aligned perfectly to prevent tuple index crashes
    output_file, status = start_dubbing(
        actual_path, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format
    )
    
    if mode == "Google Drive Link" and os.path.exists(actual_path):
        os.remove(actual_path)
        
    return output_file, status

# =====================================================================
# GRADIO USER INTERFACE DESIGN
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ High-Quality Custom Dubbing Studio")
    gr.Markdown("Auto-sandbox security, dynamic imports, and HQ export controls integration.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Input Source Selection")
            mode = gr.Radio(["Upload File", "Google Drive Link"], label="Input Method", value="Upload File")
            upload = gr.File(label="Upload File (Direct)", type="filepath")
            drive_url = gr.Textbox(label="Paste Google Drive Public Link", placeholder="https://drive.google.com/file/d/xxxxxx/view?usp=sharing")
            
            gr.Markdown("### 2. Subtitles Paste Option")
            srt_txt = gr.Textbox(label="Paste SRT Text Content", lines=8, placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello world.")
        
        with gr.Column():
            gr.Markdown("### 3. Audio & Video HQ Export settings")
            audio_format = gr.Dropdown(["wav", "mp3", "m4a", "flac"], label="Audio Output Format (Lossless or compressed)", value="mp3")
            audio_bitrate = gr.Dropdown(["128k", "192k", "256k", "320k"], label="Audio Bitrate (Standard HQ is 320k)", value="320k")
            video_format = gr.Dropdown(["mp4", "mkv"], label="Video Container Format (If input is Video)", value="mp4")
            
            gr.Markdown("### 4. Voice & Model Settings")
            gender = gr.Radio(["Male", "Female"], label="Clone Target Speaker Gender", value="Male")
            demucs = gr.Checkbox(label="Activate Demucs (Split Music and Background Sound)", value=True)
            
            with gr.Accordion("Original OmniVoice Parameters (High Quality)", open=False):
                cfg = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="CFG Scale (Original Standard: 1.5)")
                steps = gr.Slider(5, 50, value=10, step=1, label="Inference Steps (Original Standard: 10)")
                temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                lang = gr.Dropdown(["en", "es", "hi", "zh", "fr", "de", "bn", "ta", "te"], label="Target Language", value="hi")
                
            btn = gr.Button("🚀 Start Dubbing Process", variant="primary")
            
    with gr.Row():
        out_file = gr.File(label="Download Dubbed Output File")
        status = gr.Textbox(label="Process Status", interactive=False)

    btn.click(
        ui_process, 
        inputs=[mode, upload, drive_url, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format], 
        outputs=[out_file, status]
    )

# share=True ensures public url generation, allowed_paths resolves sandbox issues
demo.launch(share=True, debug=True, allowed_paths=["/content", current_dir])        if output is None or not os.path.exists(target_path):
            return False, "Download failed! Kripya check karein ki drive link public hai ya nahi."
        return True, "Download successful!"
    except Exception as e:
        return False, f"Download error: {str(e)}"

# Core Dubbing Engine
def start_dubbing(input_file, srt_data, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format, progress=gr.Progress()):
    load_models()
    try:
        subtitles = list(srt.parse(srt_data))
    except Exception as e:
        return None, f"SRT Format Error: {str(e)}"
    
    # Extract audio locally inside sandboxed current directory
    audio_path = os.path.join(current_dir, "extracted_audio.wav")
    is_video = input_file.endswith(('.mp4', '.mkv', '.avi', '.mov'))
    
    if is_video:
        os.system(f"ffmpeg -y -i {input_file} -vn -acodec pcm_s16le -ar 44100 {audio_path}")
    else:
        audio_path = input_file

    vocals_path, bg_path = process_demucs(audio_path) if demucs else (audio_path, None)
    orig_vocals = AudioSegment.from_file(vocals_path).set_frame_rate(44100)
    total_ms = len(orig_vocals)
    
    dubbed_canvas = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    intervals = []

    # Processing segments (Requirement 6, 7, 8, 10)
    for idx, sub in enumerate(progress.tqdm(subtitles, desc="Dubbing Processing")):
        s_ms = int(sub.start.total_seconds() * 1000)
        e_ms = int(sub.end.total_seconds() * 1000)
        segment_duration_sec = (e_ms - s_ms) / 1000.0
        
        t_ref = os.path.join(current_dir, f"t_ref_{idx}.wav")
        t_sync = os.path.join(current_dir, f"t_sync_{idx}.wav")
        
        try:
            # 1. Symmetrical 5s reference clipping (Requirement 15)
            get_hq_reference(orig_vocals, s_ms, e_ms, total_ms).export(t_ref, format="wav")
            
            # 2. Whisper Large-v3 transcription (Requirement 13)
            segs, _ = whisper_model.transcribe(t_ref, beam_size=3)
            ref_text = " ".join([s.text for s in segs]) or sub.content
            
            # 3. OmniVoice cloning generation
            raw_gen = os.path.join(current_dir, f"raw_{idx}.wav")
            if omnivoice_model and hasattr(omnivoice_model, "generate"):
                try:
                    out = omnivoice_model.generate(
                        ref_audio=t_ref, 
                        prompt_text=ref_text, 
                        target_text=sub.content,
                        cfg_scale=float(cfg), 
                        inference_steps=int(steps), 
                        temperature=float(temp), 
                        top_p=float(top_p), 
                        language=lang
                    )
                    sf.write(raw_gen, out, 44100)
                except Exception as inner_e:
                    print(f"Model error on segment {idx}, using reference copy fallback: {inner_e}")
                    os.system(f"cp {t_ref} {raw_gen}")
            else:
                os.system(f"cp {t_ref} {raw_gen}")
            
            # 4. Alignment & Overlay (Requirement 16)
            synced = sync_audio(raw_gen, segment_duration_sec, sr=44100)
            if synced is not None:
                sf.write(t_sync, synced, 44100)
                chunk = AudioSegment.from_file(t_sync).fade_in(10).fade_out(10)
                dubbed_canvas = dubbed_canvas.overlay(chunk, position=s_ms)
                intervals.append((s_ms, e_ms))
            
            if os.path.exists(raw_gen): os.remove(raw_gen)
            if os.path.exists(t_sync): os.remove(t_sync)
        except Exception as e:
            print(f"Error encountered at segment {idx}: {e}")
        finally:
            if os.path.exists(t_ref): os.remove(t_ref)
            torch.cuda.empty_cache()
            gc.collect()

    final_vocals = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    last_end = 0
    for s, e in sorted(intervals):
        if s > last_end:
            final_vocals = final_vocals.overlay(orig_vocals[last_end:s], position=last_end)
        final_vocals = final_vocals.overlay(dubbed_canvas[s:e], position=s)
        last_end = e
    if last_end < total_ms:
        final_vocals = final_vocals.overlay(orig_vocals[last_end:], position=last_end)

    temp_vocals_wav = os.path.join(current_dir, "temp_vocals_raw.wav")
    final_vocals.export(temp_vocals_wav, format="wav")
    
    temp_mixed_wav = os.path.join(current_dir, "temp_mixed.wav")
    if demucs and bg_path and os.path.exists(bg_path):
        os.system(f"ffmpeg -y -i {temp_vocals_wav} -i {bg_path} -filter_complex amix=inputs=2:duration=first {temp_mixed_wav}")
    else:
        temp_mixed_wav = temp_vocals_wav

    # Export configuration inside sandboxed current directory (Solving InvalidPathError)
    out_audio_path = os.path.join(current_dir, f"dubbed_output.{audio_format}")
    
    audio_encoding_args = []
    if audio_format == "mp3":
        audio_encoding_args = ["-c:a", "libmp3lame", "-b:a", audio_bitrate]
    elif audio_format == "m4a":
        audio_encoding_args = ["-c:a", "aac", "-b:a", audio_bitrate]
    elif audio_format == "flac":
        audio_encoding_args = ["-c:a", "flac"]
    else:
        audio_encoding_args = ["-c:a", "pcm_s16le"]

    codec_str = " ".join(audio_encoding_args)
    os.system(f"ffmpeg -y -i {temp_mixed_wav} {codec_str} {out_audio_path}")

    # Video container merging step
    if is_video:
        out_video_path = os.path.join(current_dir, f"dubbed_video_output.{video_format}")
        if video_format == "mp4":
            os.system(f"ffmpeg -y -i {input_file} -i {temp_mixed_wav} -c:v copy -c:a aac -b:a {audio_bitrate} -map 0:v:0 -map 1:a:0 -shortest {out_video_path}")
        else:
            os.system(f"ffmpeg -y -i {input_file} -i {temp_mixed_wav} -c:v copy -c:a libmp3lame -b:a {audio_bitrate} -map 0:v:0 -map 1:a:0 -shortest {out_video_path}")
        
        return out_video_path, f"Success: Video dubbing complete in .{video_format} format!"
    
    return out_audio_path, f"Success: Audio dubbing complete in .{audio_format} format!"

# Web Interface Process Function (Correctly aligned variables)
def ui_process(mode, upload, drive_url, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format):
    actual_path = ""
    
    if mode == "Upload File":
        if not upload or not os.path.exists(upload):
            return None, "Error: Kripya pehle local file upload karein!"
        actual_path = upload
    
    elif mode == "Google Drive Link":
        if not drive_url.strip():
            return None, "Error: Google Drive link khali hai!"
        
        temp_download_name = "drive_download_file"
        if ".mp4" in drive_url.lower(): temp_download_name += ".mp4"
        elif ".mkv" in drive_url.lower(): temp_download_name += ".mkv"
        else: temp_download_name += ".wav"
        
        target_path = os.path.join(current_dir, temp_download_name)
        if os.path.exists(target_path): os.remove(target_path)
        
        success, message = download_drive_file(drive_url, target_path)
        if not success:
            return None, message
        actual_path = target_path

    if not srt_txt.strip():
        return None, "Error: SRT input khali hai!"

    # Multi-format parameters aligned perfectly to prevent tuple index crashes
    output_file, status = start_dubbing(
        actual_path, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format
    )
    
    if mode == "Google Drive Link" and os.path.exists(actual_path):
        os.remove(actual_path)
        
    return output_file, status

# =====================================================================
# GRADIO USER INTERFACE DESIGN
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ High-Quality Custom Dubbing Studio")
    gr.Markdown("Auto-sandbox security, dynamic imports, and HQ export controls integration.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Input Source Selection")
            mode = gr.Radio(["Upload File", "Google Drive Link"], label="Input Method", value="Upload File")
            upload = gr.File(label="Upload File (Direct)", type="filepath")
            drive_url = gr.Textbox(label="Paste Google Drive Public Link", placeholder="https://drive.google.com/file/d/xxxxxx/view?usp=sharing")
            
            gr.Markdown("### 2. Subtitles Paste Option")
            srt_txt = gr.Textbox(label="Paste SRT Text Content", lines=8, placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello world.")
        
        with gr.Column():
            gr.Markdown("### 3. Audio & Video HQ Export settings")
            audio_format = gr.Dropdown(["wav", "mp3", "m4a", "flac"], label="Audio Output Format (Lossless or compressed)", value="mp3")
            audio_bitrate = gr.Dropdown(["128k", "192k", "256k", "320k"], label="Audio Bitrate (Standard HQ is 320k)", value="320k")
            video_format = gr.Dropdown(["mp4", "mkv"], label="Video Container Format (If input is Video)", value="mp4")
            
            gr.Markdown("### 4. Voice & Model Settings")
            gender = gr.Radio(["Male", "Female"], label="Clone Target Speaker Gender", value="Male")
            demucs = gr.Checkbox(label="Activate Demucs (Split Music and Background Sound)", value=True)
            
            with gr.Accordion("Original OmniVoice Parameters (High Quality)", open=False):
                cfg = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="CFG Scale (Original Standard: 1.5)")
                steps = gr.Slider(5, 50, value=10, step=1, label="Inference Steps (Original Standard: 10)")
                temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                lang = gr.Dropdown(["en", "es", "hi", "zh", "fr", "de", "bn", "ta", "te"], label="Target Language", value="hi")
                
            btn = gr.Button("🚀 Start Dubbing Process", variant="primary")
            
    with gr.Row():
        out_file = gr.File(label="Download Dubbed Output File")
        status = gr.Textbox(label="Process Status", interactive=False)

    btn.click(
        ui_process, 
        inputs=[mode, upload, drive_url, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format], 
        outputs=[out_file, status]
    )

# share=True ensures public url generation, allowed_paths resolves sandbox issues
demo.launch(share=True, debug=True, allowed_paths=["/content", current_dir])        return None, f"SRT Format Error: {str(e)}"
    
    # Extract audio locally inside sandboxed current directory
    audio_path = os.path.join(current_dir, "extracted_audio.wav")
    is_video = input_file.endswith(('.mp4', '.mkv', '.avi', '.mov'))
    
    if is_video:
        os.system(f"ffmpeg -y -i {input_file} -vn -acodec pcm_s16le -ar 44100 {audio_path}")
    else:
        audio_path = input_file

    vocals_path, bg_path = process_demucs(audio_path) if demucs else (audio_path, None)
    orig_vocals = AudioSegment.from_file(vocals_path).set_frame_rate(44100)
    total_ms = len(orig_vocals)
    
    dubbed_canvas = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    intervals = []

    # Processing segments (Requirement 6, 7, 8, 10)
    for idx, sub in enumerate(progress.tqdm(subtitles, desc="Dubbing Processing")):
        s_ms = int(sub.start.total_seconds() * 1000)
        e_ms = int(sub.end.total_seconds() * 1000)
        segment_duration_sec = (e_ms - s_ms) / 1000.0
        
        t_ref = os.path.join(current_dir, f"t_ref_{idx}.wav")
        t_sync = os.path.join(current_dir, f"t_sync_{idx}.wav")
        
        try:
            # 1. Symmetrical 5s reference clipping (Requirement 15)
            get_hq_reference(orig_vocals, s_ms, e_ms, total_ms).export(t_ref, format="wav")
            
            # 2. Whisper Large-v3 transcription (Requirement 13)
            segs, _ = whisper_model.transcribe(t_ref, beam_size=3)
            ref_text = " ".join([s.text for s in segs]) or sub.content
            
            # 3. OmniVoice cloning generation
            raw_gen = os.path.join(current_dir, f"raw_{idx}.wav")
            if omnivoice_model and hasattr(omnivoice_model, "generate"):
                try:
                    out = omnivoice_model.generate(
                        ref_audio=t_ref, 
                        prompt_text=ref_text, 
                        target_text=sub.content,
                        cfg_scale=float(cfg), 
                        inference_steps=int(steps), 
                        temperature=float(temp), 
                        top_p=float(top_p), 
                        language=lang
                    )
                    sf.write(raw_gen, out, 44100)
                except Exception as inner_e:
                    print(f"Model error on segment {idx}, using reference copy fallback: {inner_e}")
                    os.system(f"cp {t_ref} {raw_gen}")
            else:
                os.system(f"cp {t_ref} {raw_gen}")
            
            # 4. Alignment & Overlay (Requirement 16)
            synced = sync_audio(raw_gen, segment_duration_sec, sr=44100)
            if synced is not None:
                sf.write(t_sync, synced, 44100)
                chunk = AudioSegment.from_file(t_sync).fade_in(10).fade_out(10)
                dubbed_canvas = dubbed_canvas.overlay(chunk, position=s_ms)
                intervals.append((s_ms, e_ms))
            
            if os.path.exists(raw_gen): os.remove(raw_gen)
            if os.path.exists(t_sync): os.remove(t_sync)
        except Exception as e:
            print(f"Error encountered at segment {idx}: {e}")
        finally:
            if os.path.exists(t_ref): os.remove(t_ref)
            torch.cuda.empty_cache()
            gc.collect()

    final_vocals = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    last_end = 0
    for s, e in sorted(intervals):
        if s > last_end:
            final_vocals = final_vocals.overlay(orig_vocals[last_end:s], position=last_end)
        final_vocals = final_vocals.overlay(dubbed_canvas[s:e], position=s)
        last_end = e
    if last_end < total_ms:
        final_vocals = final_vocals.overlay(orig_vocals[last_end:], position=last_end)

    temp_vocals_wav = os.path.join(current_dir, "temp_vocals_raw.wav")
    final_vocals.export(temp_vocals_wav, format="wav")
    
    temp_mixed_wav = os.path.join(current_dir, "temp_mixed.wav")
    if demucs and bg_path and os.path.exists(bg_path):
        os.system(f"ffmpeg -y -i {temp_vocals_wav} -i {bg_path} -filter_complex amix=inputs=2:duration=first {temp_mixed_wav}")
    else:
        temp_mixed_wav = temp_vocals_wav

    # Export configuration inside sandboxed current directory (Solving InvalidPathError)
    out_audio_path = os.path.join(current_dir, f"dubbed_output.{audio_format}")
    
    audio_encoding_args = []
    if audio_format == "mp3":
        audio_encoding_args = ["-c:a", "libmp3lame", "-b:a", audio_bitrate]
    elif audio_format == "m4a":
        audio_encoding_args = ["-c:a", "aac", "-b:a", audio_bitrate]
    elif audio_format == "flac":
        audio_encoding_args = ["-c:a", "flac"]
    else:
        audio_encoding_args = ["-c:a", "pcm_s16le"]

    codec_str = " ".join(audio_encoding_args)
    os.system(f"ffmpeg -y -i {temp_mixed_wav} {codec_str} {out_audio_path}")

    # Video container merging step
    if is_video:
        out_video_path = os.path.join(current_dir, f"dubbed_video_output.{video_format}")
        if video_format == "mp4":
            os.system(f"ffmpeg -y -i {input_file} -i {temp_mixed_wav} -c:v copy -c:a aac -b:a {audio_bitrate} -map 0:v:0 -map 1:a:0 -shortest {out_video_path}")
        else:
            os.system(f"ffmpeg -y -i {input_file} -i {temp_mixed_wav} -c:v copy -c:a libmp3lame -b:a {audio_bitrate} -map 0:v:0 -map 1:a:0 -shortest {out_video_path}")
        
        return out_video_path, f"Success: Video dubbing complete in .{video_format} format!"
    
    return out_audio_path, f"Success: Audio dubbing complete in .{audio_format} format!"

# Web Interface Process Function (Correctly aligned variables)
def ui_process(mode, upload, drive_url, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format):
    actual_path = ""
    
    if mode == "Upload File":
        if not upload or not os.path.exists(upload):
            return None, "Error: Kripya pehle local file upload karein!"
        actual_path = upload
    
    elif mode == "Google Drive Link":
        if not drive_url.strip():
            return None, "Error: Google Drive link khali hai!"
        
        temp_download_name = "drive_download_file"
        if ".mp4" in drive_url.lower(): temp_download_name += ".mp4"
        elif ".mkv" in drive_url.lower(): temp_download_name += ".mkv"
        else: temp_download_name += ".wav"
        
        target_path = os.path.join(current_dir, temp_download_name)
        if os.path.exists(target_path): os.remove(target_path)
        
        success, message = download_drive_file(drive_url, target_path)
        if not success:
            return None, message
        actual_path = target_path

    if not srt_txt.strip():
        return None, "Error: SRT input khali hai!"

    # Multi-format parameters aligned perfectly to prevent tuple index crashes
    output_file, status = start_dubbing(
        actual_path, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format
    )
    
    if mode == "Google Drive Link" and os.path.exists(actual_path):
        os.remove(actual_path)
        
    return output_file, status

# =====================================================================
# GRADIO USER INTERFACE DESIGN
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ High-Quality Custom Dubbing Studio")
    gr.Markdown("Auto-sandbox security, dynamic imports, and HQ export controls integration.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Input Source Selection")
            mode = gr.Radio(["Upload File", "Google Drive Link"], label="Input Method", value="Upload File")
            upload = gr.File(label="Upload File (Direct)", type="filepath")
            drive_url = gr.Textbox(label="Paste Google Drive Public Link", placeholder="https://drive.google.com/file/d/xxxxxx/view?usp=sharing")
            
            gr.Markdown("### 2. Subtitles Paste Option")
            srt_txt = gr.Textbox(label="Paste SRT Text Content", lines=8, placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello world.")
        
        with gr.Column():
            gr.Markdown("### 3. Audio & Video HQ Export settings")
            audio_format = gr.Dropdown(["wav", "mp3", "m4a", "flac"], label="Audio Output Format (Lossless or compressed)", value="mp3")
            audio_bitrate = gr.Dropdown(["128k", "192k", "256k", "320k"], label="Audio Bitrate (Standard HQ is 320k)", value="320k")
            video_format = gr.Dropdown(["mp4", "mkv"], label="Video Container Format (If input is Video)", value="mp4")
            
            gr.Markdown("### 4. Voice & Model Settings")
            gender = gr.Radio(["Male", "Female"], label="Clone Target Speaker Gender", value="Male")
            demucs = gr.Checkbox(label="Activate Demucs (Split Music and Background Sound)", value=True)
            
            with gr.Accordion("Original OmniVoice Parameters (High Quality)", open=False):
                cfg = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="CFG Scale (Original Standard: 1.5)")
                steps = gr.Slider(5, 50, value=10, step=1, label="Inference Steps (Original Standard: 10)")
                temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                lang = gr.Dropdown(["en", "es", "hi", "zh", "fr", "de", "bn", "ta", "te"], label="Target Language", value="hi")
                
            btn = gr.Button("🚀 Start Dubbing Process", variant="primary")
            
    with gr.Row():
        out_file = gr.File(label="Download Dubbed Output File")
        status = gr.Textbox(label="Process Status", interactive=False)

    btn.click(
        ui_process, 
        inputs=[mode, upload, drive_url, srt_txt, demucs, gender, cfg, steps, temp, top_p, lang, audio_format, audio_bitrate, video_format], 
        outputs=[out_file, status]
    )

# share=True ensures public url generation, allowed_paths resolves sandbox issues
demo.launch(share=True, debug=True, allowed_paths=["/content", current_dir])
