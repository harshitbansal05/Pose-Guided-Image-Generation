source ~/.bashrc

if [ ! -d ./data/DF_img_pose ]; then
	if [ ! -d ./data]; then
		mkdir data
	fi
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_img_pose.zip
    unzip DF_img_pose.zip
    rm -f DF_img_pose.zip
    cd -
fi

python prepare_data.py './data/DF_img_pose' 'train'
python prepare_data.py './data/DF_img_pose' 'test'
