# Cycled training

This approach splits the dataset into smaller chunks and starts training with the first chunk
and then adds one chunk after another to the training dataset. \
It was presented by the paper
[DeepSpeech-German](https://www.researchgate.net/publication/336532830_German_End-to-end_Speech_Recognition_based_on_DeepSpeech).

### Experiment results

The experiments were run with DeepSpeech v0.6 and an old version of this project (check out the
git history if you're interested in the complete setup used for the tests). \
In difference to the paper it only had a very small effect on the accuracy,
but a high impact on training duration. Therefore, it wasn't used further.

| Dataset  | Additional Infos                                                                              | Losses           | Result                       |
| -------- | --------------------------------------------------------------------------------------------- | ---------------- | ---------------------------- |
| Voxforge | transfer learning from english, early stopping, reduce learning rate on plateau               | Test: 44.312195  | WER: 0.343973, CER: 0.140119 |
| Voxforge | 5 cycled training, rest like above                                                            | Test: 42.973358  | WER: 0.353841, CER: 0.158554 |
| Tuda     | checkpoint from Voxforge with WER 0.344, correct train/dev/test splitting                     | Test: 100.846367 | WER: 0.416950, CER: 0.198177 |
| Tuda     | 10 cycled training, checkpoint from Voxforge with WER 0.344, correct train/dev/test splitting | Test: 98.007607  | WER: 0.410520, CER: 0.194091 |

In comparison to the paper findings, the updated DS version (they used v0.5), transfer learning from English,
and the extended augmentation usage already improved the results greatly. \
Important note for the comparison: In the paper the data was randomly split. Because of the special construction of the Tuda dataset
the same transcriptions can occur in train and test set, only recorded with different microphones.
This results in a much lower WER compared to using the predefined splits of Tuda.

| Dataset | Additional Infos                                                | Result        |
| ------- | --------------------------------------------------------------- | ------------- |
| Tuda-De | DS-German paper                                                 | WER: 0.268    |
| Tuda    | random dataset split, checkpoint from Voxforge with WER 0.344   | WER: 0.090285 |
| Tuda    | test of pretrained checkpoint from DS-German with correct split | WER: 0.785655 |
