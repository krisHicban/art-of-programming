import pyaudio
import wave
import threading
from datetime import datetime
import time

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
            if info['maxInputChannels'] > 0:  # Only show input devices
                print(f"Index {i}: {info['name']}")
                print(f"  Max Input Channels: {info['maxInputChannels']}")
                print(f"  Default Sample Rate: {info['defaultSampleRate']}")
                print()
    
    def record_stream(self, device_index, filename):
        """Record from a specific device"""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        RATE = 44100
        
        # Get device info to determine channels
        device_info = self.audio.get_device_info_by_index(device_index)
        channels = min(int(device_info['maxInputChannels']), 2)
        
        if channels == 0:
            print(f"Error: Device {device_index} has no input channels!")
            return
        
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
            print(f"✓ Recording started: {filename} ({channels} channel(s))")
            
            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"Warning during recording: {e}")
                    continue
            
            stream.stop_stream()
            stream.close()
            
            # Save to file
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            print(f"✓ Finished recording: {filename}")
            
        except Exception as e:
            print(f"✗ Error recording {filename}: {e}")
    
    def record_dual(self, mic_device, speaker_device, duration=None):
        """Record from microphone and speakers simultaneously"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mic_file = f"mic_{timestamp}.wav"
        speaker_file = f"speakers_{timestamp}.wav"
        
        print("\n" + "="*50)
        print("STARTING DUAL RECORDING")
        print("="*50)
        
        self.is_recording = True
        
        # Create threads for simultaneous recording
        mic_thread = threading.Thread(
            target=self.record_stream,
            args=(mic_device, mic_file),
            name="MicThread"
        )
        
        speaker_thread = threading.Thread(
            target=self.record_stream,
            args=(speaker_device, speaker_file),
            name="SpeakerThread"
        )
        
        # Start both recordings
        mic_thread.start()
        speaker_thread.start()
        
        # Small delay to let threads start
        time.sleep(0.5)
        
        try:
            if duration:
                print(f"\n⏱  Recording for {duration} seconds...")
                time.sleep(duration)
            else:
                print("\n⏸  Press Enter to stop recording...")
                input()
        except KeyboardInterrupt:
            print("\n⏹  Interrupted! Stopping...")
        
        # Stop recording
        print("\n⏹  Stopping recording...")
        self.is_recording = False
        
        # Wait for both threads to finish
        mic_thread.join(timeout=5)
        speaker_thread.join(timeout=5)
        
        print("\n" + "="*50)
        print("RECORDINGS SAVED:")
        print("="*50)
        print(f"  🎤 Microphone: {mic_file}")
        print(f"  🔊 Speakers:   {speaker_file}")
        print("="*50 + "\n")
    
    def cleanup(self):
        """Clean up PyAudio resources"""
        self.audio.terminate()


def main():
    recorder = AudioRecorder()
    
    try:
        # List available devices
        recorder.list_devices()
        
        # Get device indices from user
        print("="*50)
        print("SELECT DEVICES")
        print("="*50)
        
        mic_index = int(input("🎤 Microphone device index: "))
        speaker_index = int(input("🔊 Speaker/Loopback device index: "))
        
        # Get duration
        duration_input = input("\n⏱  Duration in seconds (press Enter for manual stop): ")
        duration = int(duration_input) if duration_input.strip() else None
        
        # Start recording
        recorder.record_dual(mic_index, speaker_index, duration)
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        recorder.cleanup()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()