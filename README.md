# WhisperMini: Fine-Tuning Whisper for Urdu ASR

**WhisperMini** is a fine-tuned and optimized version of OpenAI’s Whisper ASR model, built to improve Urdu speech transcription in resource-constrained environments like call centers and edge devices.

## Project Highlights

* Fine-tuned the Whisper-Tiny model on Urdu data (Mozilla Common Voice)
* Word Error Rate (WER) reduced from 125.0% to 51.2%
* Outperformed the original Whisper-Base model (WER 81.0%)
* Applied dynamic quantization using HuggingFace Optimum
* Memory usage reduced by 64%, inference time improved by 30%

## Fine-Tuning Details

We fine-tuned the Whisper-Tiny model to address its poor zero-shot performance on Urdu, a low-resource language. After training:

| Model          | Original WER | Fine-Tuned WER |
| -------------- | ------------ | -------------- |
| Whisper-Tiny   | 125.0%       | 51.2%          |
| Whisper-Base   | 81.0%        | —              |
| Whisper-Medium | 39.0%        | —              |

The fine-tuned Tiny model provides a practical balance between accuracy and efficiency, making it ideal for real-world Urdu ASR applications.

## Optimization

* Applied dynamic quantization and exported the model to ONNX
* Reduced inference time from 15 seconds to 6.7 seconds (sample audio)
* Model is ready for deployment in environments with limited resources

## Model Access

The fine-tuned model is available on Hugging Face:
**[osman31/whisper-tiny-urdu-v1](https://huggingface.co/osman31/whisper-tiny-urdu-v1)**

## Usage
Just run the frontend.py file. It's a Gradio based web interface for uploading and transcribing Urdu audio from the Hugging Face model I trained.


## Future Work

* Extend to regional Urdu dialects and code-switched speech
* Explore structured pruning for further speed-up
* Pilot deployment in call center environments

## Authors

* Mohammad Osman
* Nayyera Wasim
* Fatima Ghafoor
