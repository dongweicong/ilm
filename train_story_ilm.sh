DATASET=roc_stories
TRAIN_DIR=../drive/train_story_ilm
EXAMPLES_DIR=data/char_masks/${DATASET}
python train_ilm.py \
	experiment_${DATASET} \
	${TRAIN_DIR} \
	${EXAMPLES_DIR} \
	--seed 0 \
	--wandb \
	--wandb_project_name 'story_ilm'
	--train_examples_tag train \
	--eval_examples_tag valid \
	--eval_max_num_examples 512
