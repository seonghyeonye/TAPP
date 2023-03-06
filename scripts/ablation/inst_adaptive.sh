export openai_key=""

task_dir=data/tasks
clf_file_path=clf_task_list.txt
output_dir=outputs/inst_adaptive
max_num_instances_per_eval_task=100
axis1=False
axis2=True
axis3=False
axis4=False
engine="davinci"
echo ${type}

for engine in "davinci"
do
python src/run_nearest_demo.py \
    --supernat $task_dir \
    --task_dir $task_dir \
    --clf_tasks_file $clf_file_path \
    --overwrite_cache \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1920 \
    --max_target_length 128 \
    --engine ${engine} \
    --output_dir ${output_dir} \
    --icil True \
    --axis1 ${axis1} \
    --axis2 ${axis2} \
    --axis3 ${axis3} \
    --axis4 ${axis4} \
    --adaptive
python src/compute_metrics.py --predictions ${output_dir}/predicted_examples.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics
done