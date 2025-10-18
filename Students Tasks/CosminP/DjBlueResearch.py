# -*- coding: utf-8 -*-
"""
Oracolul Digital â€“ DJ Blue Edition
- UI player-like + zonÄƒ mare de text
- Live STT (Google) + fallback Whisper
- AssemblyAI Live (EN/DE/ES/FR/IT/PT)
- Meeting: Ã®nregistrare WAV â†’ diarizare (pyannote) â†’ Whisper transcriere
- Auto-language + code-switch pe segmente (Whisper + langid dacÄƒ e instalat)
- Tuning Live + Export TXT/SRT/VTT
- VU Meter + opÈ›iune numÄƒr vorbitori

DependenÈ›e extra recomandate:
  pip install langid
"""
import os, math, tempfile, threading, tkinter as tk
from tkinter import messagebox, filedialog

# --- STT Google live
import speech_recognition as sr

# --- I/O audio
import sounddevice as sd
import soundfile as sf
import numpy as np

# --- Whisper
from faster_whisper import WhisperModel

# --- Diarizare (opÈ›ional)
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    HAS_PYANNOTE = True
except Exception:
    HAS_PYANNOTE = False

# --- LangID pentru tag per replicÄƒ (opÈ›ional)
try:
    import langid
    HAS_LANGID = True
except Exception:
    HAS_LANGID = False

# --- AssemblyAI
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient, StreamingClientOptions, StreamingParameters,
    StreamingEvents, BeginEvent, TurnEvent, TerminationEvent, StreamingError,
    StreamingSessionParameters
)

# ============ PaletÄƒ "DJ Blue â€” Neon Chill" ============
DJ_PRIMARY   = "#1E90FF"
DJ_ACCENT    = "#FF69B4"
DJ_BG        = "#0A0A12"
DJ_TEXT      = "#E0E6ED"
DJ_SECONDARY = "#6C63FF"

COLOR_BACKGROUND = DJ_BG
COLOR_PANEL      = "#131723"
COLOR_TEXT       = DJ_TEXT
COLOR_ACCENT     = DJ_PRIMARY
COLOR_BUTTON_BG  = "#22273A"
FONT_MAIN  = ("Inter", 12)
FONT_TITLE = ("Montserrat", 18, "bold")
FONT_OUTPUT= ("Space Grotesk", 11)

# ============ Device config ============
MIC_INDEX = None
SD_INPUT_DEVICE = None

# ============ Live tuning (Google) ============
LIVE_SAMPLE_RATE = 16000
LIVE_CHUNK = 1024
LIVE_ENERGY_THRESHOLD = 240
LIVE_DYNAMIC_ENERGY = True
LIVE_DYNAMIC_RATIO = 1.25
LIVE_PAUSE_THRESHOLD = 0.55
LIVE_NON_SPEAKING = 0.30
LIVE_PHRASE_THRESHOLD = 0.1
LIVE_PHRASE_TIME_LIMIT = 6

# ============ Meeting ============
MEET_SR = 16000
MEET_CHANNELS = 1
MEET_WAV_PATH = "meeting_recording.wav"

# ============ Whisper ============
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "small")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
# IMPORTANT: lÄƒsÄƒm None pentru auto (ca sÄƒ poatÄƒ schimba limba pe segmente)
WHISPER_LANGUAGE = None

# ============ Tokens ============
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
AAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")


# =====================================================================
class OracleApp:
    def __init__(self, master):
        self.master = master
        master.title("Oracolul Digital Â· DJ Blue")
        master.configure(bg=COLOR_BACKGROUND)
        master.geometry("1040x720")

        # --- Google recognizer ---
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=MIC_INDEX, sample_rate=LIVE_SAMPLE_RATE, chunk_size=LIVE_CHUNK)
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
        self.recognizer.energy_threshold = LIVE_ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = LIVE_DYNAMIC_ENERGY
        try: self.recognizer.dynamic_energy_ratio = LIVE_DYNAMIC_RATIO
        except Exception: pass
        self.recognizer.pause_threshold = LIVE_PAUSE_THRESHOLD
        self.recognizer.non_speaking_duration = LIVE_NON_SPEAKING
        self.recognizer.phrase_threshold = LIVE_PHRASE_THRESHOLD

        self.is_listening = False
        self.stop_listening_function = None

        # --- Meeting ---
        self.meet_stream = None
        self.meet_file = None
        self.meet_is_recording = False

        # --- Models lazy ---
        self.whisper_model = None
        self.pyannote_pipeline = None

        # --- AssemblyAI ---
        self.aai_client = None
        self.aai_thread = None
        self.aai_running = False

        # --- Engine selector ---
        self.live_engine = tk.StringVar(value="google")

        # --- Transcript state ---
        self.last_transcript_kind = None  # "live" / "meeting"
        self.last_dialog_lines = None     # [(speaker, start, end, text, lang)] pentru meeting

        # --- Runtime options ---
        self._live_phrase_time_limit = LIVE_PHRASE_TIME_LIMIT
        self.num_speakers_var = tk.StringVar(value="auto")  # "auto" sau int ca text
        self.auto_lang_switch = tk.BooleanVar(value=True)   # pentru meeting

        # --- VU meter ---
        self.vu_stream = None
        self.vu_level = 0.0

        # --- UI ---
        self.animation_job = None
        self.animation_frame = 0
        self._build_ui()
        self._animate_status()

    # ---------------- UI ----------------
    def _build_ui(self):
        # Header bar (player style)
        header = tk.Frame(self.master, bg=COLOR_PANEL)
        header.pack(fill="x", padx=16, pady=(16,8))
        tk.Label(header, text="Blue DJ", font=FONT_TITLE, fg=DJ_SECONDARY, bg=COLOR_PANEL).pack(side="left", padx=(10,14))

        # Engine segmented
        seg = tk.Frame(header, bg=COLOR_PANEL)
        seg.pack(side="left", padx=6)
        for name in ["google", "aai"]:
            b = tk.Radiobutton(seg, text=name.upper(), variable=self.live_engine, value=name,
                               indicatoron=0, selectcolor=DJ_PRIMARY, bg=COLOR_BUTTON_BG, fg="white",
                               padx=8, pady=4, font=("Inter", 10, "bold"), activebackground=DJ_PRIMARY)
            b.pack(side="left", padx=2)

        # Play/Stop (Live)
        self.toggle_button = tk.Button(header, text="â–¶ Live", command=self.toggle_listening,
                                       bg=DJ_PRIMARY, fg="black", font=("Inter", 11, "bold"),
                                       activebackground=DJ_ACCENT, activeforeground="black",
                                       relief="flat", padx=14, pady=6)
        self.toggle_button.pack(side="left", padx=10)

        # Meeting controls like player buttons
        self.meet_start_btn = tk.Button(header, text="âº Rec", command=self.meet_start_recording,
                                        bg=DJ_ACCENT, fg="black", font=("Inter", 11, "bold"),
                                        relief="flat", padx=14, pady=6)
        self.meet_start_btn.pack(side="left", padx=6)
        self.meet_stop_btn = tk.Button(header, text="â¹ Stop", command=self.meet_stop_and_transcribe,
                                       state="disabled", bg=COLOR_BUTTON_BG, fg="white",
                                       font=("Inter", 11, "bold"), relief="flat", padx=14, pady=6)
        self.meet_stop_btn.pack(side="left", padx=6)

        # VU meter
        self.vu_canvas = tk.Canvas(header, width=160, height=18, bg=COLOR_PANEL, highlightthickness=0)
        self.vu_canvas.pack(side="right", padx=(6,10))
        self.vu_rect = self.vu_canvas.create_rectangle(2,2,2,16, fill=DJ_PRIMARY, width=0)

        # Status
        self.status_label = tk.Label(header, text="AÈ™tept comandÄƒ", font=FONT_MAIN, fg=COLOR_TEXT, bg=COLOR_PANEL)
        self.status_label.pack(side="right", padx=10)

        # Secondary bar: tuning + options
        tools = tk.Frame(self.master, bg=COLOR_BACKGROUND)
        tools.pack(fill="x", padx=16, pady=(0,8))
        tk.Button(tools, text="âš™ï¸ Tuning Live", command=self.open_tuning_dialog,
                  bg=COLOR_BUTTON_BG, fg="white", relief="flat", padx=12, pady=6).pack(side="left", padx=4)

        tk.Label(tools, text="Speakers:", bg=COLOR_BACKGROUND, fg=COLOR_TEXT, font=FONT_MAIN).pack(side="left", padx=(18,6))
        self.spk_entry = tk.Entry(tools, textvariable=self.num_speakers_var, width=6,
                                  bg=COLOR_PANEL, fg=COLOR_TEXT, insertbackground="white", relief="flat")
        self.spk_entry.pack(side="left")
        tk.Label(tools, text="(ex: 2, 3, auto)", bg=COLOR_BACKGROUND, fg="#8aa", font=("Inter",10)).pack(side="left", padx=(6,0))

        c = tk.Checkbutton(tools, text="Auto language / code-switch", variable=self.auto_lang_switch,
                           bg=COLOR_BACKGROUND, fg=COLOR_TEXT, selectcolor=COLOR_PANEL, activebackground=COLOR_BACKGROUND)
        c.pack(side="left", padx=18)

        tk.Button(tools, text="ðŸ’¾ SalveazÄƒâ€¦", command=self.save_transcript,
                  bg=COLOR_BUTTON_BG, fg="white", relief="flat", padx=12, pady=6).pack(side="right", padx=4)

        # Main text area (big)
        self.text_output = tk.Text(self.master, height=26, width=120, font=FONT_OUTPUT,
                                   bg=COLOR_PANEL, fg=COLOR_TEXT, insertbackground="white", borderwidth=0)
        self.text_output.pack(padx=16, pady=(4,14), fill="both", expand=True)

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _animate_status(self):
        base = "Ascult" if self.is_listening else ("ÃŽnregistrare..." if self.meet_is_recording else "AÈ™tept comandÄƒ")
        dots = "." * (self.animation_frame % 4)
        self.status_label.config(text=base + dots)
        self.animation_frame += 1
        self.master.after(400, self._animate_status)
        self._draw_vu()

    # ---------------- Live master toggle ----------------
    def toggle_listening(self):
        if self.meet_is_recording:
            messagebox.showwarning("AtenÈ›ie", "OpreÈ™te Ã®ntÃ¢i Ã®nregistrarea Meeting.")
            return
        engine = self.live_engine.get()
        if self.is_listening:
            if engine == "google":
                if self.stop_listening_function: self.stop_listening_function(wait_for_stop=False)
            else:
                self._stop_aai_live()
            self.is_listening = False
            self.toggle_button.config(text="â–¶ Live", bg=DJ_PRIMARY, fg="black")
            self._stop_vu()
            return

        # start
        self.is_listening = True
        self.text_output.delete("1.0", tk.END)
        self.last_transcript_kind = "live"
        self.last_dialog_lines = None
        self.toggle_button.config(text="â¹ Stop Live", bg=DJ_SECONDARY, fg="black")
        if engine == "google":
            self.stop_listening_function = self.recognizer.listen_in_background(
                self.microphone, self._audio_callback_live_google,
                phrase_time_limit=self._live_phrase_time_limit
            )
            self.status_label.config(text="Ascult (Google)â€¦")
            self._start_vu()
        else:
            if self._start_aai_live():
                self._start_vu()
            else:
                self.is_listening = False
                self.toggle_button.config(text="â–¶ Live", bg=DJ_PRIMARY, fg="black")

    # ---- Google live + fallback Whisper
    def _audio_callback_live_google(self, recognizer, audio):
        try:
            self.master.after(0, lambda: self.status_label.config(text="Procesezâ€¦"))
            # È›inem RO pentru vitezÄƒ; fallback Whisper va prinde engleza din bucatÄƒ
            text = recognizer.recognize_google(audio, language="ro-RO")
            self.master.after(0, self._append_text, text + ". ")
        except sr.UnknownValueError:
            txt = self._whisper_chunk(audio)
            self.master.after(0, self._append_text, (txt + ". ") if txt else "[neÃ®nÈ›eles]\n")
        except sr.RequestError as e:
            self.master.after(0, self._append_text, f"[Eroare Google STT: {e}]\n")
        except Exception as e:
            self.master.after(0, self._append_text, f"[Eroare live]: {e}\n")

    def _whisper_chunk(self, audio):
        try:
            if self.whisper_model is None:
                self.whisper_model = WhisperModel(WHISPER_MODEL_NAME, compute_type=WHISPER_COMPUTE_TYPE)
            wav = audio.get_wav_data(convert_rate=16000, convert_width=2)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav); path = f.name
            # Parametri pt. code-switch (nu condiÈ›ionÄƒm prea mult pe textul anterior)
            segments, _ = self.whisper_model.transcribe(
                path,
                language=None if self.auto_lang_switch.get() else "ro",
                task="transcribe",
                beam_size=5,
                best_of=5,
                vad_filter=True,
                condition_on_previous_text=False,
                temperature=0.0,
                patience=0.2,
                prompt_reset_on_temperature=0.5
            )
            os.remove(path)
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception:
            return None

    # ---- AssemblyAI Live
    def _start_aai_live(self):
        api_key = AAI_API_KEY
        if not api_key:
            messagebox.showerror("LipseÈ™te cheia", "SeteazÄƒ ASSEMBLYAI_API_KEY.")
            return False
        self.status_label.config(text="Conectez AssemblyAIâ€¦")
        self.aai_client = StreamingClient(StreamingClientOptions(api_key=api_key, api_host="streaming.assemblyai.com"))

        def on_begin(client, e: BeginEvent):
            self.master.after(0, self._append_text, f"[AAI] Sesiune: {e.id}\n")
        def on_turn(client, e: TurnEvent):
            if e.transcript: self.master.after(0, self._append_text, e.transcript + " ")
            if e.end_of_turn and not e.turn_is_formatted:
                client.set_params(StreamingSessionParameters(format_turns=True))
        def on_term(client, e: TerminationEvent):
            self.master.after(0, self._append_text, f"\n[AAI] Terminat ({e.audio_duration_seconds}s)\n")
        def on_err(client, err: StreamingError):
            self.master.after(0, self._append_text, f"[AAI] Eroare: {err}\n")

        self.aai_client.on(StreamingEvents.Begin, on_begin)
        self.aai_client.on(StreamingEvents.Turn, on_turn)
        self.aai_client.on(StreamingEvents.Termination, on_term)
        self.aai_client.on(StreamingEvents.Error, on_err)

        def run():
            try:
                self.aai_client.connect(StreamingParameters(
                    sample_rate=16000, format_turns=True, language="multi"
                ))
                self.aai_running = True
                self.master.after(0, self.status_label.config, {"text": "Ascult (AssemblyAI)â€¦"})
                self.aai_client.stream(aai.extras.MicrophoneStream(sample_rate=16000))
            except Exception as e:
                self.master.after(0, self._append_text, f"[AAI] EÈ™ec: {e}\n")
            finally:
                try: self.aai_client.disconnect(terminate=True)
                except Exception: pass
                self.aai_running = False
                if self.is_listening and self.live_engine.get() == "aai":
                    self.is_listening = False
                    self.master.after(0, self.toggle_button.config, {"text":"â–¶ Live","bg":DJ_PRIMARY,"fg":"black"})
                    self.master.after(0, self.status_label.config, {"text":"AÈ™tept comandÄƒ"})
                self._stop_vu()
        threading.Thread(target=run, daemon=True).start()
        return True

    def _stop_aai_live(self):
        try:
            if self.aai_client: self.aai_client.disconnect(terminate=True)
        except Exception: pass
        self.aai_running = False

    # ---------------- Tuning dialog ----------------
    def open_tuning_dialog(self):
        if self.is_listening:
            messagebox.showinfo("Info", "OpreÈ™te modul live ca sÄƒ ajustezi setÄƒrile.")
            return
        win = tk.Toplevel(self.master); win.title("Tuning Live"); win.configure(bg=COLOR_PANEL); win.geometry("480x380")

        def add_scale(lbl, lo, hi, step, val, row, bag):
            frm = tk.Frame(win, bg=COLOR_PANEL); frm.grid(row=row, column=0, sticky="we", padx=12, pady=8)
            tk.Label(frm, text=lbl, bg=COLOR_PANEL, fg=COLOR_TEXT, font=FONT_MAIN).pack(anchor="w")
            sv = tk.Scale(frm, from_=lo, to=hi, resolution=step, orient="horizontal",
                          bg=COLOR_PANEL, fg=COLOR_TEXT, troughcolor=COLOR_BUTTON_BG,
                          highlightthickness=0, length=360)
            sv.set(val); sv.pack(fill="x"); bag[lbl]=sv

        box={}
        add_scale("Energy threshold", 50, 2000, 10, self.recognizer.energy_threshold, 0, box)
        add_scale("Dynamic ratio", 1.0, 2.0, 0.05, getattr(self.recognizer,"dynamic_energy_ratio",LIVE_DYNAMIC_RATIO), 1, box)
        add_scale("Pause threshold", 0.2, 1.5, 0.05, self.recognizer.pause_threshold, 2, box)
        add_scale("Non-speaking", 0.1, 1.0, 0.05, self.recognizer.non_speaking_duration, 3, box)
        add_scale("Phrase time limit", 3, 12, 1, self._live_phrase_time_limit, 4, box)

        dyn_var = tk.BooleanVar(value=self.recognizer.dynamic_energy_threshold)
        tk.Checkbutton(win, text="Dynamic energy", variable=dyn_var, bg=COLOR_PANEL, fg=COLOR_TEXT,
                       selectcolor=COLOR_BUTTON_BG).grid(row=5, column=0, sticky="w", padx=16)

        def apply_close():
            self.recognizer.energy_threshold = box["Energy threshold"].get()
            self.recognizer.dynamic_energy_threshold = dyn_var.get()
            try: self.recognizer.dynamic_energy_ratio = box["Dynamic ratio"].get()
            except Exception: pass
            self.recognizer.pause_threshold = box["Pause threshold"].get()
            self.recognizer.non_speaking_duration = box["Non-speaking"].get()
            self._live_phrase_time_limit = int(box["Phrase time limit"].get())
            messagebox.showinfo("OK", "SetÄƒrile Live au fost aplicate."); win.destroy()

        actions = tk.Frame(win, bg=COLOR_PANEL); actions.grid(row=6,column=0,sticky="e",padx=16,pady=10)
        tk.Button(actions, text="AplicÄƒ", command=apply_close, bg=COLOR_BUTTON_BG, fg="white", relief="flat", padx=12).pack(side="right", padx=6)
        tk.Button(actions, text="ÃŽnchide", command=win.destroy, bg=COLOR_BUTTON_BG, fg="white", relief="flat", padx=12).pack(side="right", padx=6)

    # ---------------- Meeting: rec + process ----------------
    def meet_start_recording(self):
        if self.is_listening:
            messagebox.showwarning("AtenÈ›ie", "OpreÈ™te modul live Ã®nainte de a Ã®ncepe Ã®nregistrarea.")
            return
        if self.meet_is_recording: return
        try:
            self.meet_file = sf.SoundFile(MEET_WAV_PATH, mode="w", samplerate=MEET_SR, channels=MEET_CHANNELS, subtype="PCM_16")
        except Exception as e:
            messagebox.showerror("Eroare", f"Nu pot crea fiÈ™ierul WAV: {e}"); return
        try:
            self.meet_stream = sd.InputStream(samplerate=MEET_SR, channels=MEET_CHANNELS, dtype="int16",
                                              device=SD_INPUT_DEVICE, callback=self._meet_audio_callback)
            self.meet_stream.start()
        except Exception as e:
            try: self.meet_file.close()
            except Exception: pass
            self.meet_file=None; messagebox.showerror("Eroare", f"Nu pot porni Ã®nregistrarea: {e}"); return

        self.meet_is_recording = True
        self.last_transcript_kind = "meeting"
        self.last_dialog_lines = None
        self._append_text("[Meeting] ÃŽnregistrare pornitÄƒ...\n")
        self.meet_start_btn.config(state="disabled")
        self.meet_stop_btn.config(state="normal", bg=DJ_SECONDARY, fg="black")
        self._start_vu()

    def _meet_audio_callback(self, indata, frames, time_info, status):
        if self.meet_file is not None:
            try: self.meet_file.write(indata.copy())
            except Exception: pass

    def meet_stop_and_transcribe(self):
        if not self.meet_is_recording: return
        # stop rec
        try: self.meet_stream.stop(); self.meet_stream.close()
        except Exception: pass
        self.meet_stream = None
        try: self.meet_file.close()
        except Exception: pass
        self.meet_file = None
        self.meet_is_recording = False
        self.meet_start_btn.config(state="normal")
        self.meet_stop_btn.config(state="disabled", bg=COLOR_BUTTON_BG, fg="white")
        self._append_text("[Meeting] ÃŽnregistrare opritÄƒ. Procesez...\n")
        self._stop_vu()
        threading.Thread(target=self._process_meeting_file, daemon=True).start()

    # ----- meeting processing -----
    def _lazy_load_models(self):
        if self.whisper_model is None:
            self._set_status("ÃŽncarc Whisper...")
            self.whisper_model = WhisperModel(WHISPER_MODEL_NAME, compute_type=WHISPER_COMPUTE_TYPE)
        if HAS_PYANNOTE and self.pyannote_pipeline is None and HF_TOKEN:
            self._set_status("ÃŽncarc diarizarea...")
            self.pyannote_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
            )

    def _process_meeting_file(self):
        try:
            self._lazy_load_models()

            # 1) Diarizare
            spk_segments = []
            if self.pyannote_pipeline is not None:
                self._set_status("Rulez diarizarea...")
                # opÈ›ional: numÄƒr de vorbitori
                ns = self.num_speakers_var.get().strip().lower()
                diar_kwargs = {}
                if ns not in ("", "auto"):
                    try:
                        diar_kwargs["num_speakers"] = int(ns)
                    except Exception:
                        pass
                diar = self.pyannote_pipeline(MEET_WAV_PATH, **diar_kwargs)
                for turn, _, speaker in diar.itertracks(yield_label=True):
                    spk_segments.append((float(turn.start), float(turn.end), speaker))
                spk_segments.sort(key=lambda x: x[0])
                # mic smoothing: lipim goluri mici < 0.2s
                merged = []
                for s,e,lab in spk_segments:
                    if not merged: merged.append([s,e,lab]); continue
                    ps,pe,pl = merged[-1]
                    if lab==pl and s-pe <= 0.2:
                        merged[-1][1] = e
                    else:
                        merged.append([s,e,lab])
                spk_segments = [tuple(x) for x in merged]

            # 2) Transcriere cu code-switch
            self._set_status("Rulez transcrierea (Whisper)...")
            segments, info = self.whisper_model.transcribe(
                MEET_WAV_PATH,
                language=None if self.auto_lang_switch.get() else "ro",
                task="transcribe",
                beam_size=5,
                best_of=5,
                vad_filter=True,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
                patience=0.2,
                prompt_reset_on_temperature=0.5
            )

            # 3) Map words -> speaker + detect limba pe segment (cu langid, dacÄƒ existÄƒ)
            def spk_at(t):
                for s,e,lab in spk_segments:
                    if s <= t < e: return lab
                return None

            dialog_lines = []
            cur_spk = None
            cur_words = []
            line_start = line_end = None

            def flush():
                if not cur_words: return
                text = " ".join(cur_words).strip()
                lang = None
                if HAS_LANGID and text and len(text) >= 6:
                    lab, prob = langid.classify(text)
                    if prob >= 0.85:
                        lang = lab
                dialog_lines.append((cur_spk, line_start, line_end, text, lang))

            for seg in segments:
                # dacÄƒ nu avem words, tratÄƒm seg Ã®ntreg
                if not seg.words:
                    mid = 0.5*(seg.start+seg.end)
                    spk = spk_at(mid) or "UNKNOWN"
                    if spk != cur_spk: flush(); cur_spk = spk; cur_words=[]; line_start = seg.start
                    cur_words.append(seg.text.strip()); line_end = seg.end
                    continue
                for w in seg.words:
                    ws,we = float(w.start), float(w.end)
                    spk = spk_at(0.5*(ws+we)) or ("SPEAKER_00" if not spk_segments else "UNKNOWN")
                    if spk != cur_spk:
                        flush(); cur_spk = spk; cur_words=[w.word]; line_start=ws; line_end=we
                    else:
                        cur_words.append(w.word); line_end=we
            flush()

            # 4) Friendly speaker names
            speaker_map, nxt = {}, 1
            def norm(label):
                nonlocal nxt
                if label not in speaker_map:
                    speaker_map[label] = f"Speaker {nxt}"; nxt += 1
                return speaker_map[label]

            # 5) Output + stocare pt export (cu tag limbÄƒ)
            self._clear_text()
            self.last_dialog_lines = []
            has_diar = bool(spk_segments)
            title = "=== Transcriere cu diarizare ===" if has_diar else "=== Transcriere (fÄƒrÄƒ diarizare) ==="
            self._append_text(title + "\n\n")

            for spk,s,e,text,lang in dialog_lines:
                name = norm(spk) if has_diar else "Speaker"
                tag = f" [{lang.upper()}]" if lang else ""
                self._append_text(f"[{self._fmt_ts(s)}â€“{self._fmt_ts(e)}] {name}{tag}: {text}\n")
                self.last_dialog_lines.append((name, s, e, text, lang))

            self.last_transcript_kind = "meeting"
            self._set_status("Gata âœ¨")
        except Exception as e:
            self._append_text(f"\n[Eroare procesare]: {e}\n")
            self._set_status("Eroare")

    # ---------------- Export ----------------
    def save_transcript(self):
        if self.last_transcript_kind == "meeting" and self.last_dialog_lines:
            types=[("SubRip (.srt)","*.srt"),("WebVTT (.vtt)","*.vtt"),("Text (.txt)","*.txt")]
            defext=".srt"
        else:
            types=[("Text (.txt)","*.txt")]; defext=".txt"
        path = filedialog.asksaveasfilename(defaultextension=defext, filetypes=types, title="SalveazÄƒ transcrierea")
        if not path: return
        try:
            if path.lower().endswith(".srt"): self._export_srt(path)
            elif path.lower().endswith(".vtt"): self._export_vtt(path)
            else: self._export_txt(path)
            messagebox.showinfo("Salvat", f"Transcriere salvatÄƒ Ã®n:\n{path}")
        except Exception as e:
            messagebox.showerror("Eroare la salvare", str(e))

    def _export_txt(self, path):
        with open(path,"w",encoding="utf-8") as f:
            f.write(self.text_output.get("1.0", tk.END).strip()+"\n")

    def _export_srt(self, path):
        if self.last_transcript_kind!="meeting" or not self.last_dialog_lines:
            return self._export_txt(path)
        def srt_ts(t):
            ms=int(round((t-int(t))*1000)); s=int(t)%60; m=(int(t)//60)%60; h=int(t)//3600
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        with open(path,"w",encoding="utf-8") as f:
            for i,(spk,s,e,text,lang) in enumerate(self.last_dialog_lines,1):
                tag=f" [{lang.upper()}]" if lang else ""
                f.write(f"{i}\n{srt_ts(s)} --> {srt_ts(e)}\n{spk}{tag}: {text}\n\n")

    def _export_vtt(self, path):
        if self.last_transcript_kind!="meeting" or not self.last_dialog_lines:
            return self._export_txt(path)
        def vtt_ts(t):
            ms=int(round((t-int(t))*1000)); s=int(t)%60; m=(int(t)//60)%60; h=int(t)//3600
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        with open(path,"w",encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for (spk,s,e,text,lang) in self.last_dialog_lines:
                tag=f" [{lang.upper()}]" if lang else ""
                f.write(f"{vtt_ts(s)} --> {vtt_ts(e)}\n{spk}{tag}: {text}\n\n")

    # ---------------- VU meter ----------------
    def _start_vu(self):
        if self.vu_stream is not None: return
        try:
            self.vu_stream = sd.InputStream(samplerate=MEET_SR, channels=1, dtype="float32",
                                            device=SD_INPUT_DEVICE, blocksize=256, callback=self._vu_callback)
            self.vu_stream.start()
        except Exception:
            self.vu_stream = None

    def _vu_callback(self, indata, frames, time_info, status):
        try:
            rms = float(np.sqrt(np.mean(np.square(indata))))
            self.vu_level = max(0.0, min(1.0, rms*8.0))
        except Exception:
            pass

    def _draw_vu(self):
        w,h=160,18
        self.vu_canvas.coords(self.vu_rect, 2, 2, 2+int((w-4)*self.vu_level), h-2)

    def _stop_vu(self):
        try:
            if self.vu_stream: self.vu_stream.stop(); self.vu_stream.close()
        except Exception: pass
        self.vu_stream=None; self.vu_level=0.0

    # ---------------- Utils ----------------
    def _set_status(self, msg): self.master.after(0, lambda: self.status_label.config(text=msg))
    def _append_text(self, txt):
        def _(): self.text_output.insert(tk.END, txt); self.text_output.see(tk.END)
        self.master.after(0, _)
    def _clear_text(self): self.master.after(0, lambda: self.text_output.delete("1.0", tk.END))
    @staticmethod
    def _fmt_ts(t):
        if t is None: return "--:--"
        m=int(t//60); s=int(t%60); return f"{m:02d}:{s:02d}"

    # ---------------- Cleanup ----------------
    def on_closing(self):
        if self.is_listening and self.stop_listening_function:
            try: self.stop_listening_function(wait_for_stop=False)
            except Exception: pass
        try:
            if self.aai_client: self.aai_client.disconnect(terminate=True)
        except Exception: pass
        try:
            if self.meet_stream: self.meet_stream.stop(); self.meet_stream.close()
        except Exception: pass
        try:
            if self.meet_file: self.meet_file.close()
        except Exception: pass
        self._stop_vu()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = OracleApp(root)
    root.mainloop()