conda create -n hseg_env python=3.9 -y
conda activate hseg_env
conda install pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=11.3.1 -c pytorch -y
conda install scipy=1.8.1 -c conda-forge -y
conda install matplotlib=3.5.2 -c conda-forge -y
conda install tensorboard=2.4.1 -c conda-forge -y
conda install ipython=8.6.0 -c anaconda -y
conda install notebook=6.5.2 -c anaconda -y
conda install pandas=1.4.4 -c anaconda -y
conda install -c conda-forge torchmetrics
conda install -c conda-forge scikit-learn  notebook tqdm ipywidgets matplotlib -y
