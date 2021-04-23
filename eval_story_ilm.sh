DATASET=roc_stories
EVAL_DIR=train_story_ilm
EXAMPLES_DIR=data/char_masks/${DATASET}
python train_ilm.py \
		eval \
		${EVAL_DIR} \
		${EXAMPLES_DIR} \
		--mask_cls ilm.mask.hierarchical.MaskHierarchical \
		--task ilm \
		--data_no_cache \
		--eval_only \
		--eval_examples_tag test \
		--eval_batch_size 4 \
		--eval_sequence_length 256 \
		--eval_skip_naive_incomplete
