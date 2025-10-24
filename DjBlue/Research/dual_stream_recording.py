# For mac, need additional driver to match pyaudio architecture
# brew install portaudio
# And reach it seems a macOS limitation in dual-recording   

# Index - 3 for speakers on PC,
# 8 - Rode Mic

import pyaudio
import wave
import threading
from datetime import datetime

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        
    def list_devices(self):
        """List all available audio devices"""
        print("\nAvailable Audio Devices:")
        print("-" * 50)
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            print(f"Index {i}: {info['name']}")
            print(f"  Max Input Channels: {info['maxInputChannels']}")
            print(f"  Default Sample Rate: {info['defaultSampleRate']}")
            print()
    
    def record_stream(self, device_index, filename, channels=2):
        """Record from a specific device"""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        RATE = 44100
        
        try:
            stream = self.audio.open(
                format=FORMAT,
                channels=channels,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            print(f"Recording to {filename}...")
            
            while self.is_recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            print(f"Finished recording {filename}")
            stream.stop_stream()
            stream.close()
            
            # Save to file
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
        except Exception as e:
            print(f"Error recording {filename}: {e}")
    
    def record_dual(self, mic_device, mix_device, duration=None):
        """Record from two devices simultaneously"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mic_file = f"mic_only_{timestamp}.wav"
        mix_file = f"full_audio_{timestamp}.wav"
        
        self.is_recording = True
        
        # Create threads for simultaneous recording
        mic_thread = threading.Thread(
            target=self.record_stream,
            args=(mic_device, mic_file, 1)  # Mono for mic
        )
        mix_thread = threading.Thread(
            target=self.record_stream,
            args=(mix_device, mix_file, 2)  # Stereo for mix
        )
        
        # Start both recordings
        mic_thread.start()
        mix_thread.start()
        
        try:
            if duration:
                print(f"Recording for {duration} seconds...")
                threading.Event().wait(duration)
            else:
                print("Press Enter to stop recording...")
                input()
        except KeyboardInterrupt:
            print("\nStopping...")
        
        self.is_recording = False
        
        # Wait for both threads to finish
        mic_thread.join()
        mix_thread.join()
        
        print(f"\nRecordings saved:")
        print(f"  - Mic only: {mic_file}")
        print(f"  - Full audio: {mix_file}")
    
    def cleanup(self):
        self.audio.terminate()


# Example usage
if __name__ == "__main__":
    recorder = AudioRecorder()
    
    # Step 1: List devices to find the right indices
    recorder.list_devices()
    
    # Step 2: Set your device indices
    # You need to identify:
    # - Your microphone device index
    # - Your stereo mix/loopback device index
    
    print("\nEnter device indices:")
    mic_index = int(input("Microphone device index: "))
    mix_index = int(input("Stereo mix/loopback device index: "))
    
    # Step 3: Record
    duration = input("Duration in seconds (press Enter for manual stop): ")
    duration = int(duration) if duration else None
    
    recorder.record_dual(mic_index, mix_index, duration)
    recorder.cleanup()