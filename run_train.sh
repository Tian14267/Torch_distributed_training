#:<<EOF
####  单卡训练
#CUDA_VISIBLE_DEVICES="0" python run_fffan_single_gpu.py

#####   多卡训练  DataParallel
#CUDA_VISIBLE_DEVICES="3,4"  python run_fffan_mutilGPU.py

####  多卡训练  --master_port 为可选项。当同一台机器中有多个多卡训练任务时，需要用这个参数进行区分
#CUDA_VISIBLE_DEVICES="3,4"  python -m torch.distributed.launch --nproc_per_node=2  --master_port=29501 run_fffan_distributed.py --n_gpu 2
#EOF

#CUDA_VISIBLE_DEVICES="3,4"  python -m torch.distributed.launch --nproc_per_node=2  --master_port=29501 run_fffan_apex_distributed.py --n_gpu 2

####  多卡训练  horovod 加速
CUDA_VISIBLE_DEVICES="1,2" horovodrun -np 2 python run_fffan_horovod_distributed.py
