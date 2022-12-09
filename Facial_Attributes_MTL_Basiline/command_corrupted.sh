# python3 train.py --gpu 0 --filename 40tasks_resnet18_pt --pretrained > log_40tasks_resnet18_pt.txt & 
# python3 train.py --gpu 1 --filename 40tasks_resnet18 > log_40tasks_resnet18.txt & 

# python3 train.py --gpu 2 --filename 9tasks_resnet18_pt --pretrained > log_9tasks_resnet18_pt.txt & 
# python3 train.py --gpu 3 --filename 9tasks_resnet18 > log_9tasks_resnet18.txt & 

# python3 train.py --gpu 3 --lr 0.005 --filename 9tasks_resnet18_5xlr > log_9tasks_resnet18_5xlr.txt &


python3 train.py --gpu 0 --model lenet --filename 9tasks_lenet_5xlr --lr 0.005 > log_9tasks_lenet_5xlr.txt &
python3 train.py --gpu 1 --model lenet --filename 9tasks_lenet  > log_9tasks_lenet.txt &
python3 train_afrm.py --gpu 2 --model lenet --filename 9tasks_lenet_5xlr_afrm --lr 0.005 > log_9tasks_lenet_5xlr_afrm.txt &
python3 train_afrm.py --gpu 3 --model lenet --filename 9tasks_lenet_afrm  > log_9tasks_lenet_afrm.txt &
python3 train.py --gpu 4 --model alexnet --filename 9tasks_alexnet_5xlr --lr 0.005 > log_9tasks_alexnet_5xlr.txt &
python3 train_afrm.py --gpu 5 --model alexnet --filename 9tasks_alexnet_5xlr_afrm --lr 0.005 > log_9tasks_alexnet_5xlr_afrm.txt &
python3 train_afrm.py --gpu 6 --filename 9tasks_resnet18_pt_5xlr --pretrained --lr 0.005 > log_9tasks_resnet18_pt_5xlr.txt &

# python3 train.py --gpu 2 --model alexnet --filename 9tasks_alexnet > log_9tasks_alex.txt & 

# python3 train_afrm.py --gpu 0 --filename 9tasks_resnet18_afrm > log_9tasks_resnet18_afrm.txt &
# python3 train_afrm.py --gpu 1 --lr 0.005 --filename 9tasks_resnet18_afrm_5xlr > log_9tasks_resnet18_afrm_5xlr.txt &

