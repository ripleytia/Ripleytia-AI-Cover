"""Compatibility metadata and download URLs for Roformer models missing from audio-separator 0.18."""

import os

CUSTOM_ROFORMER_MODELS = {
    "Roformer Model: MelBand Roformer | Vocals by becruily": {
        "mel_band_roformer_vocals_becruily.ckpt": "config_mel_band_roformer_vocals_becruily.yaml",
    },
    "Roformer Model: MelBand Roformer | Instrumental by becruily": {
        "mel_band_roformer_instrumental_becruily.ckpt": "config_mel_band_roformer_instrumental_becruily.yaml",
    },
    "Roformer Model: Mel-Roformer-Karaoke-Aufr33-Viperx": {
        "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml",
    },
    "Roformer Model: MelBand Roformer | De-Reverb by anvuew": {
        "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt": "dereverb_mel_band_roformer_anvuew.yaml",
    },
    "Roformer Model: MelBand Roformer | Vocals by Kimberley Jensen": {
        "vocals_mel_band_roformer.ckpt": "vocals_mel_band_roformer.yaml",
    },
    "Roformer Model: MelBand Roformer | Karaoke by Gabox": {
        "mel_band_roformer_karaoke_gabox.ckpt": "config_mel_band_roformer_karaoke_gabox.yaml",
    },
    "Roformer Model: BS-Roformer-De-Reverb": {
        "deverb_bs_roformer_8_384dim_10depth.ckpt": "deverb_bs_roformer_8_384dim_10depth_config.yaml",
    },
}

CUSTOM_ROFORMER_DOWNLOAD_URLS = {
    "vocals_mel_band_roformer.ckpt": "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/vocals_mel_band_roformer.ckpt",
    "vocals_mel_band_roformer.yaml": "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/vocals_mel_band_roformer.yaml",
    "mel_band_roformer_karaoke_gabox.ckpt": "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/mel_band_roformer_karaoke_gabox.ckpt",
    "config_mel_band_roformer_karaoke_gabox.yaml": "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/config_mel_band_roformer_karaoke_gabox.yaml",
    "deverb_bs_roformer_8_384dim_10depth.ckpt": "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/deverb_bs_roformer_8_384dim_10depth.ckpt",
    "deverb_bs_roformer_8_384dim_10depth_config.yaml": "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/deverb_bs_roformer_8_384dim_10depth_config.yaml",
}

CUSTOM_CONFIG_TO_CHECKPOINT = {
    config: checkpoint
    for model_files in CUSTOM_ROFORMER_MODELS.values()
    for checkpoint, config in model_files.items()
}


def patch_model_catalog(Separator):
    """Add current public Roformer models to older audio-separator registries."""
    if not getattr(Separator, "_ripleytia_roformer_catalog_patched", False):
        original_list_supported_model_files = Separator.list_supported_model_files

        def list_supported_model_files(self):
            model_files = original_list_supported_model_files(self)
            mdxc_models = model_files.setdefault("MDXC", {})
            for friendly_name, model_files_by_name in CUSTOM_ROFORMER_MODELS.items():
                mdxc_models.setdefault(friendly_name, dict(model_files_by_name))
            return model_files

        Separator.list_supported_model_files = list_supported_model_files
        Separator._ripleytia_roformer_catalog_patched = True

    if not getattr(Separator, "_ripleytia_roformer_downloads_patched", False):
        original_download_model_files = Separator.download_model_files

        def download_model_files(self, model_filename):
            checkpoint = CUSTOM_CONFIG_TO_CHECKPOINT.get(model_filename, model_filename)
            config_filename = None
            for model_files in CUSTOM_ROFORMER_MODELS.values():
                if checkpoint in model_files:
                    config_filename = model_files[checkpoint]
                    break

            if checkpoint in CUSTOM_ROFORMER_DOWNLOAD_URLS and config_filename:
                for filename in (checkpoint, config_filename):
                    url = CUSTOM_ROFORMER_DOWNLOAD_URLS.get(filename)
                    if url:
                        output_path = os.path.join(self.model_file_dir, filename)
                        self.download_file_if_not_exists(url, output_path)

            return original_download_model_files(self, model_filename)

        Separator.download_model_files = download_model_files
        Separator._ripleytia_roformer_downloads_patched = True