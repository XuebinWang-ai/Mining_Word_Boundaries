#! /bin/bash

folder=AISHELL2
folder=ZX
DATAP=../Data/$folder/

char_dict=Documents/mandarin_mfa.ext.dict

char_acoustic_model=Documents/mandarin_mfa.zip

output_dir=aligned_result/$folder

mkdir -p $output_dir
# using char dict
type='adapt'
type='align'

log_file=log/$type.$folder.log

nohup mfa $type $DATAP $char_dict $char_acoustic_model $output_dir \
-t temp/ \
--config_path config.yaml \
--verbose \
-j 6 \
> $log_file 2>&1 &


