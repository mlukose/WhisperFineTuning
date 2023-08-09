from datasets import load_dataset, DatasetDict
from transformers import Seq2SeqTrainer, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, WhisperConfig
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import os

# Hugging Face Token: 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
# Username: Micsterlukose

output_directory = 'C:\\Users\\micst\\Desktop\\Whisper Fine Tuning\\outputs\\Malayalam'

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    




def main():
    print("Hello")
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=False)
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=False)
    
    # Uses only 2000 waveforms instead of the full data set
    #common_voice["train"] = common_voice["train"].select(range(200))
    #common_voice["test"] = common_voice["test"].select(range(200))

    #print(common_voice)

    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

    #print(common_voice)

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="malayalam", task="transcribe")

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="malayalam", task="transcribe")

    configuration = WhisperConfig(dropout=0.1).from_pretrained("openai/whisper-small", language="malayalam", task="transcribe")

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)
    
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    model = WhisperForConditionalGeneration(config=configuration).from_pretrained("openai/whisper-small").to(device=device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []


    training_args = Seq2SeqTrainingArguments(
        output_dir=output_directory,  # change to a repo name of your choice
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # increase by 2x for every 2x decrease in batch size
        learning_rate=7.7e-6,
        warmup_steps=0,
        max_steps=1200,
        gradient_checkpointing=False,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        resume_from_checkpoint="C:\\Users\\micst\\Desktop\\Whisper Fine Tuning\\outputs\\Malayalam\\1300_Complete-3",
    )


    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    
    
    trainer.train()

    processor.save_pretrained(training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    config_json_file = os.path.join(training_args.output_dir, "config.json")
    model.config.to_json_file(config_json_file)

if __name__ == '__main__':
    main()