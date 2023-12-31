# Empty Colab default directory
rm -rf sample_data
# Empty any previous clone of repo
rm -rf scriptable_hubert_encoder
# Clone repo and move files to root
git clone https://github.com/fauxneticien/scriptable_hubert_encoder.git
cp -r scriptable_hubert_encoder/* .
rm -rf scriptable_hubert_encoder
# Install dependencies
pip install torch==2.0.1 torchaudio==2.0.2 lhotse==1.16.0 gdown
apt-get update && apt-get install -y tree
# Get mini-librispeech data
gdown 1KOwmYkPS5UFKNFDBI7rTbuBrzr_xrRBW
mkdir -p data/mini-librispeech
tar -xzf mini-librispeech.tgz -C data/mini-librispeech
