export openai_key=""

data_dir=data/splits/default
task_dir=data/tasks
output_dir=demos/analysis/relaxed
demo_path=demos/ICIL/ICIL_seed1.json

max_num_instances_per_eval_task=100
type="Demo Extraction with Relaxed Sampling" 

echo ${type}
# 0, 42, 123, 10, 20, 30, 40, 50, 60, 70
for seed in 0 42 123 10 20 30 40 50 60 70
do
echo $engine
python preprocess/relaxed_sampling.py \
    --supernat ${task_dir} \
    --random_seed ${seed} \
    --data_dir ${data_dir} \
    --output_dir ${output_dir} 
done