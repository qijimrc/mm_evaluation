#!/bin/bash

data_root=$(dirname $(dirname $(dirname $(dirname $(pwd)))))/data/VQAv2

# Download
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P $data_root && unzip $data_root/v2_Questions_Val_mscoco.zip -d $data_root
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P $data_root && unzip $data_root/v2_Annotations_Val_mscoco.zip -d $data_root

# Clear cache
rm $data_root/v2_Questions_Val_mscoco.zip $data_root/v2_Annotations_Val_mscoco.zip