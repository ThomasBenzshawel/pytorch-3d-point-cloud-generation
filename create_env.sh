# Script to create the environment!
# THis may work once we get a better yml file
# conda env create -f env.yml

# For now, run each of these commands sequentially to make the environment
conda env create --name tommy
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tensorboardX
conda install matplotlib scipy pandas
pip3 install Imath OpenEXR
conda deactivate
