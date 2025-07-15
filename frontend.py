import os
import sys
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import gradio as gr

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Set environment variables for ffmpeg
if sys.platform == "win32":
    ffmpeg_path = r"C:\Users\Me\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
    if os.path.exists(ffmpeg_path):
        os.environ["FFMPEG_BINARY"] = ffmpeg_path
        print(f"Found ffmpeg at: {ffmpeg_path}")
    else:
        print("Warning: Could not find ffmpeg. Please ensure it's installed and in PATH")

# Initialize both pipelines with GPU support
print("Loading Urdu fine-tuned model...")
urdu_pipe = pipeline(
    task="automatic-speech-recognition",
    model="Osman31/whisper-tiny-urdu-v1",
    chunk_length_s=30,
    stride_length_s=5,
    device=device,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    generate_kwargs={"language": "urdu", "task": "transcribe"}
)

print("Loading original Whisper model...")
original_pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-tiny",
    chunk_length_s=30,
    stride_length_s=5,
    device=device,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    generate_kwargs={"language": "urdu", "task": "transcribe"}
)

print("Models loaded successfully!")

def transcribe(audio):
    if audio is None:
        return "Please provide an audio file.", "Please provide an audio file."
    
    try:
        # Clear GPU cache before inference if using CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Get transcriptions from both models
        print("Transcribing with Urdu fine-tuned model...")
        urdu_text = urdu_pipe(audio)["text"]
        
        print("Transcribing with original Whisper model...")
        original_text = original_pipe(audio)["text"]
        
        # Clear GPU cache after inference
        if device == "cuda":
            torch.cuda.empty_cache()
            
        return urdu_text, original_text
    except Exception as e:
        error_msg = f"Error during transcription: {str(e)}"
        print(error_msg)
        return error_msg, error_msg

# Create the Gradio interface
iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs=[
        gr.Textbox(label="Urdu Fine-tuned Model Transcription", text_align="right"),
        gr.Textbox(label="Original Whisper-tiny Transcription", text_align="right")
    ],
    title="Whisper Model Comparison (GPU Accelerated)",
    description=f"Compare Urdu transcriptions between the fine-tuned Whisper tiny model and the original Whisper tiny model. Running on: {device.upper()}",
    examples=None,
    cache_examples=False
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()