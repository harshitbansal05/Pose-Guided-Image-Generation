source ~/.bashrc

if [ ! -d ./data/DF_img_pose/DF_train_data ]; then
    ./run_prepare_data.sh
fi


#######################################################################
################################ Training #############################
gpu=0

python train_model.py --img_H=256  --img_W=256 \
             --batch_size=1 --max_step=80000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --is_train=True \
             --gpu=${gpu} \
             --z_num=64 \
