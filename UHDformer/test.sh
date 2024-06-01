python setup.py develop
CUDA_VISIBLE_DEVICES=0 python inference_uhdformer.py
#CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4398 inference_femasr_psnr_ssim_ychannel.py --launcher pytorch