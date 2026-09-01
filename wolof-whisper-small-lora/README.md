---
license: mit
datasets:
- M9and2M/Wolof_ASR_dataset
language:
- wo
metrics:
- wer
pipeline_tag: automatic-speech-recognition
---

# Wolof ASR Model (Based on Whisper-Small)

## Model Overview

This repository hosts an Automatic Speech Recognition (ASR) model for the Wolof language, fine-tuned from OpenAI's Whisper-small model. This model aims to provide accurate transcription of Wolof audio data.

## Model Details

- **Model Base**: Whisper-small
- **Loss**: 0.123
- **WER**: 0.17

## Dataset

The dataset used for training and evaluating this model is a collection from various sources, ensuring a rich and diverse set of Wolof audio samples. The collection is available in my Hugging Face account is used by keeping only the audios with duration shorter than 6 second.

- **Training Dataset**: 57 hours
- **Test Dataset**: 10 hours

For detailed information about the dataset, please refer to the [M9and2M/Wolof_ASR_dataset](https://huggingface.co/datasets/M9and2M/Wolof_ASR_dataset).

## Training

The training process was adapted from the code in the [Finetune Wa2vec 2.0 For Speech Recognition](https://github.com/khanld/ASR-Wa2vec-Finetune) written to fine-tune Wav2Vec2.0 for speech recognition. Special thanks to the author, Duy Khanh, Le for providing a robust and flexible training framework.

The model was trained with the following configuration:

- **Seed**: 19
- **Training Batch Size**: 1
- **Gradient Accumulation Steps**: 8
- **Number of GPUs**: 2

### Optimizer : AdamW

- **Learning Rate**: 1e-7

### Scheduler: OneCycleLR

- **Max Learning Rate**: 5e-5

## Acknowledgements
This model was built using OpenAI's [Whisper-small](https://huggingface.co/openai/whisper-small) architecture and fine-tuned with a dataset collected from various sources. Special thanks to the creators and contributors of the dataset.


<!-- ## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

<!-- **BibTeX:** -->

<!-- [More Information Needed] -->

<!-- **APA:** -->



## More Information

This model has been developed in the context of my Master Thesis at ETSIT-UPM, Madrid under the supervision of Prof. Luis A. Hernández Gómez.


## Contact

For any inquiries or questions, please contact mamadou.marone@ensea.fr