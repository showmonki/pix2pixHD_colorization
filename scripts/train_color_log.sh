nohup python train.py --name color --no_instance --dataroot datasets/video_color --model colorization --niter 1 --niter_decay 1  > logfiles/test-`date +%Y%m%d-%H%M`.log 2>&1 ; /usr/bin/shutdown
