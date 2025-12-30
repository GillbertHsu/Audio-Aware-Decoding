import json
import librosa
from numba import none
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, LogitsProcessor
from tqdm import tqdm
import re
import numpy as np
from datetime import datetime
import gc

torch.manual_seed(42)

use_logits_processor = True  # Control whether to use the logits processor
alpha = 0.5
filename = "sample.json"

prefix_prompt = "Focus on the given audio and answer the following question. "

batch_size = 6
max_new_tokens = 256  # For generation


'''
Logits processor is the key component for AAD, it 
modifies the logits of the next token to be generated
during the generation process
'''

class AudioLogitsProcessor(LogitsProcessor):
    def __init__(self, model, embeds_without_audio, atts_without_audio, alpha=0.5):
        """
        Args:
            model: The language model
            embeds_without_audio: Initial embeddings with zeroed audio portion
            atts_without_audio: Attention mask for no-audio input
            alpha: Scaling factor for context-aware decoding
        """
        self.model = model
        self.alpha = alpha
        
        # Store initial no-audio embeddings
        self.embeds_without_audio = embeds_without_audio
        self.atts_without_audio = atts_without_audio
        self.first_call = True
        
    def __call__(self, input_ids, scores):
        """
        Implements: yt ∼ softmax[(1 + α)logit_θ(yt|c,x,y<t) - αlogit_θ(yt|x,y<t)]
        where c is audio context, x is text context, y<t is previously generated tokens
        
        Args:
            input_ids: Current sequence including generated tokens
            scores: logit_θ(yt|c,x,y<t) - logits from main generation (with audio)
        """
        
        with torch.no_grad():
            if not self.first_call:
                # Only append new token embeddings after the first call
                # Get embeddings for the newly generated token
                new_token = input_ids[:, -1:]  # Get last token 
                new_token_embeds = self.model.get_input_embeddings()(new_token)
                
                # Append new token embeddings to no-audio sequence
                self.embeds_without_audio = torch.cat([self.embeds_without_audio, new_token_embeds], dim=1)
                
                # Update attention mask
                new_attention = torch.ones_like(new_token, dtype=self.atts_without_audio.dtype)
                self.atts_without_audio = torch.cat([self.atts_without_audio, new_attention], dim=1)
            else:
                # First call, no need to append anything
                self.first_call = False
            
            # Get logits for the same sequence but without audio
            outputs_without_audio = self.model(
                inputs_embeds=self.embeds_without_audio,
                attention_mask=self.atts_without_audio
            )
            logits_without_audio = outputs_without_audio.logits[:, -1, :]  # Get logits for next token
        
        # Apply context-aware formula
        modified_logits = (1 + self.alpha) * scores - self.alpha * logits_without_audio
        
        return modified_logits

def extract_yes_no(text):
    # Define a regular expression pattern to find "yes" or "no"
    pattern = r'\b(yes|no)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(0).lower()
    
    elif "there is no sound" in text or "there is no sound of" in text or "there is no" in text or "is not" in text:
        return "no"

    elif "does not contain" in text or "doesn't contain" in text:
        return "no"
    
    elif "contain" in text or "contains" in text:
        return "yes"
    
    elif "not" in text or "unable" in text or "can't" in text:
        return "no"
    else:
        return ""
    

def discriminative_metric(result):
    '''
    we view "no" as the positive class
    since we are measuring hallucination, 
    we want model to output "no" when it didn't hear the sound
    '''
    acc = 0
    precision = 0
    recall = 0
    f1 = 0
    yes_count = 0
    yes_true_positives = 0  # Initialize this variable
    no_pred_count = 0
    true_positives = 0
    total_actual_no = 0
    total_actual_yes = 0
    not_answer_count = 0
    
    for res in result:
        # check for audio understanding
        if res['yes_no'] == "":
            not_answer_count += 1
            continue
            
        # 1) Accuracy is unchanged:
        if res['yes_no'] == res['label']:
            acc += 1
            
        # Count yes responses for yes_ratio calculation
        if res['yes_no'] == 'yes':
            yes_count += 1
            if res['label'] == 'yes':
                yes_true_positives += 1  # Now this will work since it's initialized
            
        # 2) Count every "no" prediction:
        if res['yes_no'] == 'no':
            no_pred_count += 1
            if res['label'] == 'no':
                true_positives += 1
                
        # 3) Count every actual "no" label:
        if res['label'] == 'no':
            total_actual_no += 1
        if res['label'] == 'yes':
            total_actual_yes += 1

    # Calculate metrics
    total = len(result)
    acc = acc / total if total > 0 else 0
    
    # precision = TP / (# predicted "yes")
    precision = true_positives / no_pred_count if no_pred_count > 0 else 0
    
    # recall = TP / (# actual "yes")
    recall = true_positives / total_actual_no if total_actual_no > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
        
    # yes ratio is the ratio of yes responses 
    yes_ratio = yes_count / total if total > 0 else 0
    
    # not_answer_rate is the ratio of empty responses
    not_answer_rate = not_answer_count / total if total > 0 else 0

    print(f"Accuracy: {round(acc, 3)}, Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, F1: {round(f1, 3)}, Yes_ratio: {round(yes_ratio, 3)}
    
    return round(acc, 3), round(precision, 3), round(recall, 3), round(f1, 3), round(yes_ratio, 3)

def load_audio(qa_item, sampling_rate):
    """Load and process a single audio file"""
    try:
        audio_path = qa_item["path"]
        audio, sr = librosa.load(
            audio_path, 
            sr=sampling_rate,
            mono=True
        )
        zero_audio = np.zeros_like(audio)
        if sr != sampling_rate:
            print(f"Warning: Audio file {audio_path} was resampled from {sr} to {sampling_rate} Hz")
        
        
        return {
            "audio": audio,
            "zero_audio": zero_audio,
            "qa_item": qa_item,
            "success": True
        }
    except Exception as e:
        print(f"Error loading audio file {qa_item['path']}: {str(e)}")
        return {"success": False}

def process_batch(batch_data, model, processor):
    """Process a batch of data with the model"""
    
    sampling_rate = processor.feature_extractor.sampling_rate
    batch_results = []
    
    try:
        # Unpack the batch data
        valid_batch, audios, valid_texts, zero_audios = batch_data
        
        # Prepare inputs
        inputs = processor(text=valid_texts, audio=audios, return_tensors="pt", padding=True, sampling_rate=sampling_rate)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Only prepare no-audio inputs if using logits processor
        if use_logits_processor:
            inputs_without_audio = processor(text=valid_texts, audio=None, return_tensors="pt", padding=True, sampling_rate=sampling_rate)
            # you can also put zero_audios here
            # inputs_without_audio = processor(text=valid_texts, audio=zero_audios, return_tensors="pt", padding=True, sampling_rate=sampling_rate)
            inputs_without_audio = {k: v.to(device) for k, v in inputs_without_audio.items()}
            
            # Get input embeddings directly from the embedding layer
            input_embeddings = model.get_input_embeddings()(inputs_without_audio["input_ids"])
            logits_processor = AudioLogitsProcessor(model, input_embeddings, inputs_without_audio["attention_mask"], alpha=alpha)
            logits_processor_list = [logits_processor]
        else:
            logits_processor_list = []
        
        # Generate
        with torch.no_grad():  # Ensure we're not tracking gradients for inference
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                length_penalty=1.0,
                early_stopping=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                logits_processor=logits_processor_list,
                temperature=None,
                top_p=None,
                top_k=None
            )
            generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

        # Decode responses
        responses = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Process results
        for qa_item, response in zip(valid_batch, responses):
            result_item = {
                "yes_no": extract_yes_no(response),
                "label": qa_item["text"].lower(),
                "response": response  # Keep the full response for analysis
            }
            batch_results.append(result_item)
        
        # Clean up GPU memory
        del inputs
        if use_logits_processor:
            del inputs_without_audio
            del input_embeddings
            del logits_processor
        del generate_ids
        torch.cuda.empty_cache()
        
        return batch_results
        
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return []

def main():
    print(f"Starting evaluation sequentially.")
    print(f"Processing file: {filename}")
    
    # Load test data
    with open(filename, 'r') as f:
        qa_pairs = json.load(f)
    
    # Initialize model and processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    
    # Load single model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", 
        device_map=device,
        torch_dtype=torch.float16
    )
    model.tie_weights()
    
    all_results = []
    
    # Create progress bar for overall processing
    total_items = len(qa_pairs)
    pbar = tqdm(total=total_items, desc="Processing QA pairs")
    processed_count = 0
    
    # Process batches sequentially
    for i in range(0, len(qa_pairs), batch_size):
        batch = qa_pairs[i:min(i + batch_size, len(qa_pairs))]
        
        # Convert to conversation format
        conversations = [
            [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": qa_item["path"]},
                    {"type": "text", "text": prefix_prompt + qa_item["Q"]}
                ]
            }] for qa_item in batch
        ]
        
        # Prepare text inputs
        texts = [
            processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]
        
        # Load audio files sequentially
        sampling_rate = processor.feature_extractor.sampling_rate
        audios = []
        zero_audios = []
        valid_batch = []
        valid_texts = []
        
        for idx, qa_item in enumerate(batch):
            result = load_audio(qa_item, sampling_rate)
            if result["success"]:
                audios.append(result["audio"])
                valid_batch.append(result["qa_item"])
                valid_texts.append(texts[idx])
                if use_logits_processor:
                    zero_audios.append(np.zeros_like(result["audio"]))
        
        if not audios:
            continue
        
        # Process this batch
        batch_data = (valid_batch, audios, valid_texts, zero_audios)
        results = process_batch(batch_data, model, processor)
        all_results.extend(results)
        processed_count += len(results)
        pbar.update(len(results))
        
        # Clean up memory after each batch
        gc.collect()
        torch.cuda.empty_cache()
    
    pbar.close()
    
    print("\nCalculating metrics...")
    # Calculate and display metrics
    acc, precision, recall, f1, yes_ratio = discriminative_metric(all_results)
    
    # Create benchmark results dictionary
    benchmark_results = {
        "metrics": {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "yes_ratio": yes_ratio
        }
    }
    
    # Save detailed results with benchmark metrics
    final_results = {
        "benchmark_metrics": benchmark_results,
        "data_set": filename,
        "use_cad": use_logits_processor,
        "cad_alpha": alpha,
        "prefix_prompt": prefix_prompt,        
        "processed_items": processed_count,
        "total_items": total_items
    }
    
    # Date and time
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Save all results
    result_filename = f"evaluation_results_{date_time}.json"
    with open(result_filename, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nProcessed {processed_count} items out of {total_items} total items")
    print(f"Results saved to {result_filename}")

if __name__ == "__main__":
    main()
