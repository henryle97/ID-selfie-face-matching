pip install -r requirements.txt

npm install localtunnel

cd center/models/
rm -rf DCNv2
git clone https://github.com/CharlesShang/DCNv2.git
cd DCNv2
sh make.sh
python testcuda.py
cd ../../../

wget https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0 -O model-r100-arcface-ms1m-refine-v2.zip
unzip model-r100-arcface-ms1m-refine-v2.zip -d weights
rm model-r100-arcface-ms1m-refine-v2.zip

gdown https://drive.google.com/uc?id=1lubOiOXsh9A4D5FUCKubaR1K8GSfnnVq
mv model_cmnd_best.pth weights