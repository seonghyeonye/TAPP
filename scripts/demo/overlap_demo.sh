export openai_key=""

data_dir=data/splits/default
task_dir=data/tasks
output_dir=demos/analysis/answer_choice_overlaps
clf_file_path=clf_task_list.txt

max_num_instances_per_eval_task=100
type="Demo Extraction with 100% overlapping answer (i.e. yes, no) set" 

echo ${type}
python preprocess/overlap_sampling.py \
    --supernat ${task_dir} \
    --data_dir ${data_dir} \
    --output_dir ${output_dir} \
    --clf_tasks_file ${clf_file_path}