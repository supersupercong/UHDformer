#sleep 7h 40m 55s
python setup.py develop
#CUDA_VISIBLE_DEVICES=0,1,2,3 python basicsr/train.py -opt options/train_FeMaSR_LQ_stage.yml
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4398 basicsr/train.py -opt options/train_uhdformer.yml --launcher pytorch