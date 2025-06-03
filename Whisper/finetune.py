import argparse
from datasets import load_dataset, DatasetDict, Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import soundfile as sf

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # load dataset
    dataset = DatasetDict()

    dataset['train'] = load_dataset(f'formospeech/{args.dataset}_asr_aligned', split='train')
    dataset['test'] = load_dataset(f'formospeech/{args.dataset}_asr_aligned', split='test')

    dataset = dataset.remove_columns(['id', 'stem', 'duration'])

    speakers = [
        "XM006", "XM007", "XM008", "XM009", "XM010", "XM011", "XM012", "XM013", "XM014", "XM015",
        "XM016", "XM017", "XM038", "XM021", "XM022", "XM023", "XM025", "XM027", "XM028", "XM029",
        "XM030", "XM031", "XM032", "XM033", 
            
        "XF005", "XF006","XF007", "XF008", "XF010", "XF011", "XF012", "XF013", "XF014", "XF015", 
        "XF016", "XF018","XF019", "XF020", "XF021", "XF022", "XF023", "XF024", "XF025", "XF027", 
        "XF028", "XF029","XF030", "XF031"
    ]
    speakers_set = set(speakers)

    if args.dataset == 'hat':
        if args.exp_name == 'topline':
            dataset['target'] = dataset['train'].filter(lambda x: x['channel'] == 'webcam' and x['speaker'] in speakers_set)
        else:
            dataset['source'] = dataset['train'].filter(lambda x: x['channel'] == 'condenser' and x['speaker'] in speakers_set)
        dataset['test'] = dataset['test'].filter(lambda x: x['channel'] == 'webcam')
    else:
        if args.exp_name == 'topline':
            dataset['target'] = dataset['train'].filter(lambda x: x['channel'] == 'android')
        else:
            dataset['source'] = dataset['train'].filter(lambda x: x['channel'] == 'condenser')
        dataset['test'] = dataset['test'].filter(lambda x: x['channel'] == 'android')

    def update_audio_files(dataset, generated_audio_dir):
        def update_audio(example):
            new_path = f"{generated_audio_dir}/{example['audio']['path']}"
            example['audio']['path'] = new_path
            new_array, _ = sf.read(new_path)
            example['audio']['array'] = new_array
            return example

        return dataset.map(update_audio, num_proc=4)
    
    if args.exp_name == 'vanilla':
        pass
    elif args.exp_name == 'topline':
        dataset['target'] = update_audio_files(dataset['target'], args.topline_train_audio_dir)
    else:
        dataset['source'] = update_audio_files(dataset['source'], args.generated_train_audio_dir)

    dataset['test'] = update_audio_files(dataset['test'], args.test_audio_dir)

    # load the processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-tiny')
    tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-tiny', task='transcribe')
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny', task='transcribe')

    # data preprocessing
    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_features'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
        batch['labels'] = tokenizer(batch['text']).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names['train'], num_proc=4)

    # load the model
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny').to(device)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{'input_features': feature['input_features']} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')
            label_features = [{'input_ids': feature['labels']} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')
            labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch['labels'] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # evaluation metric
    metric = evaluate.load('cer')

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        cer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {'cer': cer}

    if args.exp_name == 'topline':
        train_dataset = dataset['target']
    else:
        train_dataset = dataset['source']
    dataset_size = len(train_dataset)
    steps_per_epoch = dataset_size // args.train_batch_size
    max_steps = steps_per_epoch * args.num_epochs
    warmup_steps = max_steps // 10
    logging_steps = max_steps // 200

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=64//args.train_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy='steps',
        per_device_eval_batch_size=args.test_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=steps_per_epoch,
        eval_steps=steps_per_epoch,
        logging_steps=logging_steps,
        report_to=['tensorboard'],
        load_best_model_at_end=True,
        metric_for_best_model='cer',
        greater_is_better=False,
        push_to_hub=True,
    )

    # train the model
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dataset['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    # push the model to Hugging Face Hub
    kwargs = {
        'dataset_tags': f'formospeech/{args.dataset}_asr_aligned',
        'dataset': f'{args.dataset.upper()} ASR Aligned',
        'dataset_args': f'config: {args.language}, split: test',
        'language': 'zh',
        'model_name': f'Whisper Tiny {args.language.capitalize()} ({args.exp_name})',
        'finetuned_from': 'openai/whisper-tiny',
        'tasks': 'automatic-speech-recognition',
    }
    trainer.push_to_hub(**kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--generated_train_audio_dir', type=str, required=True)
    parser.add_argument('--topline_train_audio_dir', type=str, required=True)
    parser.add_argument('--test_audio_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
