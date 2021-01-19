import nemo.collections.asr as nemo_asr
import numpy as np
import onnxruntime
import torch
import tqdm
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.metrics.wer import WER

# ==================================================================================================

test_manifest = "/dsp_nemo/data/test.json"

# Make sure you convert the transcriptions to lower case after exporting them with corcua
# test_manifest = "/data_prepared/en/librispeech/nm_dev-clean.json"


# ==================================================================================================


def to_numpy(tensor):
    if tensor.requires_grad:
        nt = tensor.detach().cpu().numpy()
    else:
        nt = tensor.cpu().numpy()
    return nt


# ==================================================================================================


def setup_transcribe_dataloader(cfg, vocabulary):
    config = {
        "manifest_filepath": test_manifest,
        "sample_rate": 16000,
        "labels": vocabulary,
        "batch_size": cfg["batch_size"],
        "trim_silence": True,
        "shuffle": False,
    }
    dataset = AudioToCharDataset(
        manifest_filepath=config["manifest_filepath"],
        labels=config["labels"],
        sample_rate=config["sample_rate"],
        int_values=config.get("int_values", False),
        augmentor=None,
        max_duration=config.get("max_duration", None),
        min_duration=config.get("min_duration", None),
        max_utts=config.get("max_utts", 0),
        blank_index=config.get("blank_index", -1),
        unk_index=config.get("unk_index", -1),
        normalize=config.get("normalize_transcripts", False),
        trim=config.get("trim_silence", True),
        load_audio=config.get("load_audio", True),
        parser=config.get("parser", "en"),
        add_misc=config.get("add_misc", False),
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        collate_fn=dataset.collate_fn,
        drop_last=config.get("drop_last", False),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )


# ==================================================================================================


def print_input(nemo_model_name):
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(
        model_name=nemo_model_name
    )
    config = {"batch_size": 1}
    temporary_datalayer = setup_transcribe_dataloader(
        config, quartznet.decoder.vocabulary
    )
    np.set_printoptions(edgeitems=10)

    for test_batch in temporary_datalayer:
        processed_signal, processed_signal_len = quartznet.preprocessor(
            input_signal=test_batch[0].to(quartznet.device),
            length=test_batch[1].to(quartznet.device),
        )

        nps = to_numpy(processed_signal)
        nps = np.transpose(nps, [0, 2, 1])
        print(nps)
        print(nps.shape)
        break


# ==================================================================================================


def run_full_test(nemo_model_name, batch_size, print_predictions):

    config = {"batch_size": batch_size}
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(
        model_name=nemo_model_name
    )
    ort_session = onnxruntime.InferenceSession(
        "/dsp_nemo/models/" + nemo_model_name + ".onnx"
    )
    temporary_datalayer = setup_transcribe_dataloader(
        config, quartznet.decoder.vocabulary
    )
    wer_calc = WER(
        vocabulary=quartznet.decoder.vocabulary,
        batch_dim_index=0,
        use_cer=False,
        ctc_decode=True,
    )

    wer_nums = []
    wer_denoms = []

    for test_batch in tqdm.tqdm(temporary_datalayer):
        processed_signal, processed_signal_len = quartznet.preprocessor(
            input_signal=test_batch[0].to(quartznet.device),
            length=test_batch[1].to(quartznet.device),
        )
        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(processed_signal),
        }
        ologits = ort_session.run(None, ort_inputs)
        alogits = np.asarray(ologits)
        logits = torch.from_numpy(alogits[0])
        greedy_predictions = logits.argmax(dim=-1, keepdim=False)

        if print_predictions:
            hypotheses = wer_calc.ctc_decoder_predictions_tensor(greedy_predictions)
            print(hypotheses)

        targets = test_batch[2]
        targets_lengths = test_batch[3]
        quartznet._wer.update(greedy_predictions, targets, targets_lengths)
        _, wer_num, wer_denom = quartznet._wer.compute()
        wer_nums.append(wer_num.detach().cpu().numpy())
        wer_denoms.append(wer_denom.detach().cpu().numpy())

    # We need to sum all numerators and denominators first. Then divide.
    print(f"WER = {sum(wer_nums) / sum(wer_denoms)}")


# ==================================================================================================

print_input("QuartzNet5x5LS-En")
# run_full_test("QuartzNet5x5LS-En", batch_size=16, print_predictions=False)
