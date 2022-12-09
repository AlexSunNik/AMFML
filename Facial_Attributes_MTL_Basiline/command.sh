python3 train.py --gpu 0 --filename 9tasks_convnet_cosine --model convnet --lr 0.01 --nepoch 15 --lr-scheduler cosine > 9tasks_convnet_cosine.txt &
python3 train_afrm.py --gpu 1 --filename 9tasks_convnet_cosine_afrm --model convnet --lr 0.01 --nepoch 15 --lr-scheduler cosine > 9tasks_convnet_cosine_afrm.txt &
python3 train_afrm.py --gpu 2 --filename 9tasks_lenet_cosine_afrm_correct --model lenet --lr 0.01 --nepoch 15 --lr-scheduler cosine > 9tasks_lenet_cosine_afrm_correct.txt &

# python3 train.py --gpu 0 --filename 9tasks_resnet18_cosine --lr 0.01 --nepoch 15 --lr-scheduler cosine > 9tasks_resnet18_cosine.txt &
# python3 train_afrm.py --gpu 1 --filename 9tasks_resnet18_cosine_afrm --lr 0.01 --nepoch 15 --lr-scheduler cosine > 9tasks_resnet18_cosine_afrm.txt &

# python3 train.py --gpu 2 --filename 9tasks_alexnet_cosine --model alexnet --lr 0.01 --nepoch 15 --lr-scheduler cosine > 9tasks_alexnet_cosine.txt &
# python3 train_afrm.py --gpu 3 --filename 9tasks_alexnet_cosine_afrm --model alexnet --lr 0.01 --nepoch 15 --lr-scheduler cosine > 9tasks_alexnet_cosine_afrm.txt &

# python3 train.py --gpu 4 --filename 9tasks_lenet_cosine --model lenet --lr 0.01 --nepoch 15 --lr-scheduler cosine > 9tasks_lenet_cosine.txt &
# python3 train_afrm.py --gpu 5 --filename 9tasks_lenet_cosine_afrm --model lenet --lr 0.01 --nepoch 15 --lr-scheduler cosine > 9tasks_lenet_cosine_afrm.txt &

# python3 train.py --gpu 6 --filename 9tasks_resnet18_cosine_pt --pretrained --lr 0.01 --nepoch 15 --lr-scheduler cosine > 9tasks_resnet18_cosine_pt.txt &

