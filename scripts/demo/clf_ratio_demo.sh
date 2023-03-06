export openai_key=""

data_dir=data/splits/default
task_dir=data/tasks
output_dir=demos/analysis/clf_ratio
demo_path=demos/ICIL/ICIL_seed1.json

max_num_instances_per_eval_task=100
type="Demo Extraction with various classification task ratio" 

echo ${type}

for ratio in 0 0.25 0.5 0.75 1
do
echo $engine
python preprocess/clf_sampling.py \
    --supernat ${task_dir} \
    --mode "random" \
    --output_dir ${output_dir} \
    --ratio ${ratio} \
    --demo_path ${demo_path}
done