## Use MFA to align speech-text parallel data.

### Data and File Structure
Take AISHELL2 as an example.

- AISHELL2
    - C0001
        - IC0001W0001.lab
        - IC0001W0001.wav
        - ...
    - C0002
    - ...

The `.lab` file is the text corresponding to the `.wav` file.
In the text, each Chinese character is separated by a space.


### Alignment

You can find more information at [Montreal Forced Aligner Website](https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html).

#### Installation

```
conda create -n mfa python=3.10
conda config --add channels conda-forge
conda install montreal-forced-aligner
```

#### Pronunciation dictionariy and Acoustic model

Use [Mandarin MFA acoustic model v2_0_0](https://mfa-models.readthedocs.io/en/latest/acoustic/Mandarin/Mandarin%20MFA%20acoustic%20model%20v2_0_0.html#Mandarin%20MFA%20acoustic%20model%20v2_0_0) with its corresponding [Pronunciation dictionariy](https://mfa-models.readthedocs.io/en/latest/dictionary/Mandarin/Mandarin%20MFA%20dictionary%20v2_0_0.html#mandarin-mfa-dictionary-v2-0-0) to align.

In order to do character-level alignment, we extended the pronunciation dictionary by add more adding pronunciations for polyphonetic characters.
Place the acoustic model and pronunciation dictionary in the `MFA-align/align-scripts/Documents/` folder.

You can find more acoustic models for Mandarin [here](https://mfa-models.readthedocs.io/en/latest/acoustic/Mandarin/index.html).

You can find more pronunciation dictionaries for Mandarin [hera](https://mfa-models.readthedocs.io/en/latest/dictionary/Mandarin/index.html).


#### Script

Run MFA aligner on the dataset. 
```
nohup mfa $type $DATAP $char_dict $char_acoustic_model $output_dir \
    -t temp/ \
    --config_path config.yaml \
    --verbose \
    -j 6 \
    > $log_file 2>&1 &
```
or
```
cd align-scipts/
bash mfa_align.sh
```
See more details in the `MFA-align/align-scripts/mfa_align.sh`.

