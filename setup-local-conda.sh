# Install dependencies
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install lhotse==1.16.0 hydra-core==1.3.2 gdown==4.7.1
# Get mini-librispeech data
gdown 1KOwmYkPS5UFKNFDBI7rTbuBrzr_xrRBW
tar -xzf mini-librispeech.tgz -C data/mini-librispeech
