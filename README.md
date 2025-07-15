# WhisperMini: Fine-Tuning Whisper for Urdu ASR

**WhisperMini** is a fine-tuned and optimized version of OpenAI’s Whisper ASR model, designed to improve Urdu speech transcription, especially in resource-constrained environments like call centers and mobile devices.

## 🔍 Project Highlights

* 📉 **Fine-Tuned Whisper-Tiny** on Urdu data (Mozilla Common Voice)
* ⚡ **WER Reduced** from 125.0% ➝ **51.2%**
* 🧠 Outperformed the original Whisper-Base (81.0% WER)
* 🧊 Applied **dynamic quantization** using HuggingFace Optimum
* 💾 **Memory usage cut by 64%**, and **inference time reduced by 30%**

## 🧪 Fine-Tuning Details

We fine-tuned the Whisper-Tiny model on Urdu speech data to address Whisper's poor zero-shot performance on low-resource languages. The original model struggled with a 125% Word Error Rate (WER). After fine-tuning:

| Model          | Original WER | Fine-Tuned WER |
| -------------- | ------------ | -------------- |
| Whisper-Tiny   | 125.0%       | 51.2%          |
| Whisper-Base   | 81.0%        | —              |
| Whisper-Medium | 39.0%        | —              |

The fine-tuned tiny model became significantly more usable and even outperformed larger base models, making it ideal for real-world Urdu ASR deployment.

## ⚙️ Optimization

* **Quantization** via dynamic quantization + ONNX export.
* Optimized models are **platform-independent** and suitable for edge deployment.
* Inference time reduced from 15s ➝ 6.7s for sample audio.

## Usage

Gradio based web interface for uploading and transcribing Urdu audio.

## 📌 Future Work

* Support for dialects & Urdu-English code-switching
* Pruning & kernel-level optimizations
* Real-world deployment evaluation (e.g., call center)

## 👥 Authors

* Mohammad Osman
* Nayyera Wasim
* Fatima Ghafoor
