CUDA_VISIBLE_DEVICES=0 python3 train_basenet_afrm.py --dataset nyuv2 --network_name nyu_seg_sn_afrm &
CUDA_VISIBLE_DEVICES=1 python3 train_basenet_afrm.py --dataset nyuv2 --network_name nyu_seg_sn_afrm_1e_m4 --afrm-reg --afrm-regcoef 0.0001 &
CUDA_VISIBLE_DEVICES=2 python3 train_basenet_afrm.py --dataset nyuv2 --network_name nyu_seg_sn_afrm_1e_m3 --afrm-reg --afrm-regcoef 0.001 &
CUDA_VISIBLE_DEVICES=3 python3 train_basenet_afrm.py --dataset nyuv2 --network_name nyu_seg_sn_afrm_1e_m2 --afrm-reg --afrm-regcoef 0.01 & 
CUDA_VISIBLE_DEVICES=4 python3 train_basenet_afrm.py --dataset nyuv2 --network_name nyu_seg_sn_afrm_1e_m1 --afrm-reg --afrm-regcoef 0.1 & 
CUDA_VISIBLE_DEVICES=5 python3 train_basenet.py --dataset nyuv2 --network_name nyu_seg_sn &
