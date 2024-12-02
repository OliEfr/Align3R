mkdir -p PointOdyssey
cd PointOdyssey
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partaa
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partab
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partac
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partad
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/val.tar.gz

cat train.tar.gz.partaa train.tar.gz.partab.1 train.tar.gz.partac train.tar.gz.partad > train.tar.gz 
tar -zxvf train.tar.gz 
tar -zxvf val.tar.gz