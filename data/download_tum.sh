mkdir -p tum
cd tum
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_halfsphere.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_rpy.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_halfsphere.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_rpy.tgz

for file in *.tgz; do
    if [ -e "$file" ]; then
        tar -xvzf "$file"
    else
        echo "no find .tgz"
    fi
done