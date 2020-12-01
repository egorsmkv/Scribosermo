data_dir = 'data'

# ==================================================================================================

# import glob
# import os
# import subprocess
# import tarfile
# import wget
#
# # Download the dataset. This will take a few moments...
# print("******")
# if not os.path.exists(data_dir + '/an4_sphere.tar.gz'):
#     an4_url = 'http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz'
#     an4_path = wget.download(an4_url, data_dir)
#     print(f"Dataset downloaded at: {an4_path}")
# else:
#     print("Tarfile already exists.")
#     an4_path = data_dir + '/an4_sphere.tar.gz'
#
# if not os.path.exists(data_dir + '/an4/'):
#     # Untar and convert .sph to .wav (using sox)
#     tar = tarfile.open(an4_path)
#     tar.extractall(path=data_dir)
#
#     print("Converting .sph to .wav...")
#     sph_list = glob.glob(data_dir + '/an4/**/*.sph', recursive=True)
#     for sph_path in sph_list:
#         wav_path = sph_path[:-4] + '.wav'
#         cmd = ["sox", sph_path, wav_path]
#         subprocess.run(cmd)
# print("Finished conversion.\n******")

# ==================================================================================================

# import librosa
#
# # Load and listen to the audio file
# example_file = data_dir + '/an4/wav/an4_clstk/mgah/cen2-mgah-b.wav'
# audio, sample_rate = librosa.load(example_file)
#
# import librosa.display
# import matplotlib.pyplot as plt
#
# # Plot our example audio file's waveform
# plt.rcParams['figure.figsize'] = (15,7)
# plt.title('Waveform of Audio Example')
# plt.ylabel('Amplitude')
#
# _ = librosa.display.waveplot(audio)
# plt.show()
# plt.clf()
#
# import numpy as np
#
# # Get spectrogram using Librosa's Short-Time Fourier Transform (stft)
# spec = np.abs(librosa.stft(audio))
# spec_db = librosa.amplitude_to_db(spec, ref=np.max)  # Decibels
#
# # Use log scale to view frequencies
# librosa.display.specshow(spec_db, y_axis='log', x_axis='time')
# plt.colorbar()
# plt.title('Audio Spectrogram')
# plt.show()
# plt.clf()
#
# # Plot the mel spectrogram of our sample
# mel_spec = librosa.feature.melspectrogram(audio, sr=sample_rate)
# mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#
# librosa.display.specshow(
#     mel_spec_db, x_axis='time', y_axis='mel')
# plt.colorbar()
# plt.title('Mel Spectrogram')
# plt.show()
# plt.clf()

# ==================================================================================================

# # NeMo's "core" package
# import nemo
# # NeMo's ASR collection - this collections contains complete ASR models and
# # building blocks (modules) for ASR
# import nemo.collections.asr as nemo_asr
#
# # This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
# quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
#
# files = [data_dir + '/an4/wav/an4_clstk/mgah/cen2-mgah-b.wav']
# for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
#   print(f"Audio in {fname} was recognized as: {transcription}")

# ==================================================================================================

# import json
# import os
# import librosa
#
#
# # Function to build a manifest
# def build_manifest(transcripts_path, manifest_path, wav_path):
#   with open(transcripts_path, 'r') as fin:
#     with open(manifest_path, 'w') as fout:
#       for line in fin:
#         # Lines look like this:
#         # <s> transcript </s> (fileID)
#         transcript = line[: line.find('(') - 1].lower()
#         transcript = transcript.replace('<s>', '').replace('</s>', '')
#         transcript = transcript.strip()
#
#         file_id = line[line.find('(') + 1: -2]  # e.g. "cen4-fash-b"
#         audio_path = os.path.join(
#           data_dir, wav_path,
#           file_id[file_id.find('-') + 1: file_id.rfind('-')],
#           file_id + '.wav')
#
#         duration = librosa.core.get_duration(filename=audio_path)
#
#         # Write the metadata to the manifest
#         metadata = {
#           "audio_filepath": audio_path,
#           "duration": duration,
#           "text": transcript
#         }
#         json.dump(metadata, fout)
#         fout.write('\n')
#
#
# # Building Manifests
# print("******")
# train_transcripts = data_dir + '/an4/etc/an4_train.transcription'
# train_manifest = data_dir + '/an4/train_manifest.json'
# if not os.path.isfile(train_manifest):
#   build_manifest(train_transcripts, train_manifest, 'an4/wav/an4_clstk')
#   print("Training manifest created.")
#
# test_transcripts = data_dir + '/an4/etc/an4_test.transcription'
# test_manifest = data_dir + '/an4/test_manifest.json'
# if not os.path.isfile(test_manifest):
#   build_manifest(test_transcripts, test_manifest, 'an4/wav/an4test_clstk')
#   print("Test manifest created.")
# print("***Done***")

# ==================================================================================================


import nemo.collections.asr as nemo_asr
try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML
config_path = './NeMo/examples/asr/conf/config.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
print(params)

train_manifest = data_dir + '/an4/train_manifest.json'
test_manifest = data_dir + '/an4/test_manifest.json'

# Training
import pytorch_lightning as pl
trainer = pl.Trainer(gpus=1, max_epochs=50)

from omegaconf import DictConfig
params['model']['train_ds']['manifest_filepath'] = train_manifest
params['model']['validation_ds']['manifest_filepath'] = test_manifest
first_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

import copy
new_opt = copy.deepcopy(params['model']['optim'])
new_opt['lr'] = 0.001
first_asr_model.setup_optimization(optim_config=DictConfig(new_opt))

# Start training!!!
# trainer.fit(first_asr_model)
# first_asr_model.save_to(data_dir + "/ckpt.nemo")

# ==================================================================================================

# # Bigger batch-size = bigger throughput
# params['model']['validation_ds']['batch_size'] = 16
#
# # Setup the test data loader and make sure the model is on GPU
# first_asr_model.setup_test_data(test_data_config=params['model']['validation_ds'])
# first_asr_model.cuda()
#
# # We will be computing Word Error Rate (WER) metric between our hypothesis and predictions.
# # WER is computed as numerator/denominator.
# # We'll gather all the test batches' numerators and denominators.
# wer_nums = []
# wer_denoms = []
#
# # Loop over all test batches.
# # Iterating over the model's `test_dataloader` will give us:
# # (audio_signal, audio_signal_length, transcript_tokens, transcript_length)
# # See the AudioToCharDataset for more details.
# for test_batch in first_asr_model.test_dataloader():
#         test_batch = [x.cuda() for x in test_batch]
#         targets = test_batch[2]
#         targets_lengths = test_batch[3]
#         log_probs, encoded_len, greedy_predictions = first_asr_model(
#             input_signal=test_batch[0], input_signal_length=test_batch[1]
#         )
#         # Notice the model has a helper object to compute WER
#         first_asr_model._wer.update(greedy_predictions, targets, targets_lengths)
#         _, wer_num, wer_denom = first_asr_model._wer.compute()
#         wer_nums.append(wer_num.detach().cpu().numpy())
#         wer_denoms.append(wer_denom.detach().cpu().numpy())
#
# # We need to sum all numerators and denominators first. Then divide.
# print(f"WER = {sum(wer_nums)/sum(wer_denoms)}")

# ==================================================================================================

train_manifest = '/home/isse/RoboJack/data_prepared/de/voxforge/manifest_train_azce.json'
val_manifest = '/home/isse/RoboJack/data_prepared/de/voxforge/manifest_dev_azce.json'
test_manifest = '/home/isse/RoboJack/data_prepared/de/voxforge/manifest_test_azce.json'
params['model']['train_ds']['manifest_filepath'] = train_manifest
params['model']['validation_ds']['manifest_filepath'] = val_manifest

params['model']['train_ds']['batch_size'] = 16
params['model']['validation_ds']['batch_size'] = 16

# Transfer
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

# Use the smaller learning rate we set before
quartznet.setup_optimization(optim_config=DictConfig(new_opt))

# Point to the data we'll use for fine-tuning as the training set
quartznet.setup_training_data(train_data_config=params['model']['train_ds'])

# Point to the new validation data for fine-tuning
quartznet.setup_validation_data(val_data_config=params['model']['validation_ds'])

# And now we can create a PyTorch Lightning trainer and call `fit` again.
trainer = pl.Trainer(gpus=[1], max_epochs=10)
trainer.fit(quartznet)

# ==================================================================================================

# Bigger batch-size = bigger throughput
params['model']['validation_ds']['batch_size'] = 8

# Setup the test data loader and make sure the model is on GPU
# params['model']['validation_ds']['manifest_filepath'] = test_manifest
quartznet.setup_test_data(test_data_config=params['model']['validation_ds'])
quartznet.cuda()

# We will be computing Word Error Rate (WER) metric between our hypothesis and predictions.
# WER is computed as numerator/denominator.
# We'll gather all the test batches' numerators and denominators.
wer_nums = []
wer_denoms = []

# Loop over all test batches.
# Iterating over the model's `test_dataloader` will give us:
# (audio_signal, audio_signal_length, transcript_tokens, transcript_length)
# See the AudioToCharDataset for more details.
for test_batch in quartznet.test_dataloader():
        test_batch = [x.cuda() for x in test_batch]
        targets = test_batch[2]
        targets_lengths = test_batch[3]
        log_probs, encoded_len, greedy_predictions = quartznet(
            input_signal=test_batch[0], input_signal_length=test_batch[1]
        )
        # Notice the model has a helper object to compute WER
        quartznet._wer.update(greedy_predictions, targets, targets_lengths)
        _, wer_num, wer_denom = quartznet._wer.compute()
        wer_nums.append(wer_num.detach().cpu().numpy())
        wer_denoms.append(wer_denom.detach().cpu().numpy())

# We need to sum all numerators and denominators first. Then divide.
print(f"WER = {sum(wer_nums)/sum(wer_denoms)}")
