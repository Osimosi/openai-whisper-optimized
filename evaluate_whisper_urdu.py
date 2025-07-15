import pandas as pd
from datasets import Dataset, DatasetDict, Audio
import evaluate
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from tqdm import tqdm
import os

def load_urdu_dataset(data_dir):
    """
    Load Urdu dataset from local TSV files
    data_dir: Directory containing train.tsv, dev.tsv, test.tsv and clips folder
    """
    # Load TSV files
    test_df = pd.read_csv(os.path.join(data_dir, 'test.tsv'), sep='\t')
    
    # Add full audio path
    clips_dir = os.path.join(data_dir, 'clips')
    test_df['audio'] = test_df['path'].apply(lambda x: os.path.join(clips_dir, x))
    
    # Keep only necessary columns
    columns_to_keep = ['audio', 'sentence']
    test_df = test_df[columns_to_keep]
    
    # Convert to Hugging Face dataset
    test_dataset = Dataset.from_pandas(test_df)
    
    # Cast the audio column to audio format with correct sampling rate
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return test_dataset

def main():
    # Load the original whisper tiny model and processor
    model_id = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Set up model generation config for Urdu
    model.generation_config.language = "urdu"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    # Load Urdu test dataset from local files
    data_dir = "S:/cv-corpus-20.0-2024-12-06/ur"  # Your local data directory
    test_dataset = load_urdu_dataset(data_dir)
    
    # Load WER metric
    wer_metric = evaluate.load("wer")
    
    # Lists to store predictions and references
    predictions = []
    references = []
    
    # Process each audio file
    for item in tqdm(test_dataset):
        # Get audio array
        audio = item["audio"]
        
        # Process audio
        input_features = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"], 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
            
        # Decode prediction
        transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        # Store prediction and reference
        predictions.append(transcription)
        references.append(item["sentence"])
        
    # Calculate WER
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    
    print(f"\nResults for {model_id} on Urdu test set (Common Voice 20.0):")
    print(f"Word Error Rate: {wer:.2f}%")
    
    # Save results to file
    with open("whisper_urdu_evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Dataset: Common Voice 20.0\n")
        f.write(f"Word Error Rate: {wer:.2f}%\n\n")
        f.write("Sample Predictions:\n")
        for ref, pred in zip(references[:10], predictions[:10]):
            f.write(f"\nReference: {ref}\n")
            f.write(f"Prediction: {pred}\n")
            f.write("-" * 50)

if __name__ == "__main__":
    main() 