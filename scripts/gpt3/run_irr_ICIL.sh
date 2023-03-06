export openai_key=""

data_dir=data/splits/default
task_dir=data/tasks
demo_path=demos/irr_ICIL/irr_ICIL_seed1.json
output_dir=output/irr_ICIL_davinci 
max_num_instances_per_eval_task=100

type="irrelevant ICIL" 
echo ${type}

for engine in "davinci"
do
echo $engine
python src/run_gpt3.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --overwrite_cache \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 2048 \
    --max_target_length 128 \
    --engine ${engine} \
    --output_dir ${output_dir} \
    --icil True \
    --adaptive True \
    --demo_path ${demo_path} \
    --irrelevant True \
    --cc_news_path /workspace/KAIRI/Self-Correct/SuperNat/cc_news_text_sents.json
python src/compute_metrics.py --predictions ${output_dir}/predicted_examples.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics
done