# WhisperFineTuning

Open AI Whisper is a speech to text (STT) algorithm. It maps spoken speech from an audio file recording (e.g. WAV or MP3) to text characters in the spoken language. Whisper is pre-trained on a vast quantity of labelled audio-transcription data (680,000 hours) from 96 languages. 
The majority of the data was used to train English so it is no surprise that the model is very good at transcribing english. However there are many other langauges that didn't get nearly as much training as english did. My interest was to evaluate Whisper WER performance on Malayalam which is the mother tongue of my parents.
Malayalam is one of many low resourced languages  where the Whisper zero-shot WER performance is poor (~60%). My goal in this project was to fine-tune Whisper model using additional Malayalam training audio-transcription data to improve WER.
