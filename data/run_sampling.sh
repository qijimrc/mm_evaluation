#~/bin/bash
N=${1:-500}
SEED=${2:-9271}

basedir=/nxchinamobile2/shared/qiji/repos/mm_evaluation/data
cd $basedir

programs=()
args=()

# VQAv2
vqav2_questions=/nxchinamobile2/shared/instruction_data/MultiInstruct/raw/VQA_V2/v2_OpenEnded_mscoco_val2014_questions.json
vqav2_anns=/nxchinamobile2/shared/instruction_data/MultiInstruct/raw/VQA_V2/v2_mscoco_val2014_annotations.json
save_file=/nxchinamobile2/shared/instruction_data/evaluation/vqav2_sampled.json
programs+=("process_vqav2.py")
args+=("--vqav2_question_file $vqav2_questions --vqav2_ann_file $vqav2_anns --N $N --seed $SEED --save_file $save_file")


# Process
for ((i=0; i<${#programs[@]}; i++)); do
    echo "processing ${programs[$i]} ..."
    python ${programs[$i]} ${args[$i]}
done