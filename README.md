The recommended setup steps are as follows:

1. **Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)** (and [mamba](https://github.com/mamba-org/mamba); recommended)

2. **create the conda environment**:
```bash
cd clvision-challenge-2022
conda env create -f environment.yml
```

3. **Download and extract the dataset**: in order to download the dataset, we ask all participants to accept the dataset terms and provide their email addresses through [this form](https://forms.gle/kwB3CRKAxkiJWVQ57). 
You will immediately receive the download instructions at the provided address. We recommend extracting the dataset in the default folder `$HOME/3rd_clvision_challenge/demo_dataset/`.
The final directory structure should be like this:

```
$HOME/3rd_clvision_challenge/challenge/
├── ego_objects_challenge_test.json
├── ego_objects_challenge_train.json
├── images
│   ├── 07A28C4666133270E9D65BAB3BCBB094_0.png
│   ├── 07A28C4666133270E9D65BAB3BCBB094_100.png
│   ├── 07A28C4666133270E9D65BAB3BCBB094_101.png
│   ├── ...
```

4. **change the dataset path**:
change the dataset path in the ```detection.py``` file.
```python 
# TODO: change this to the path where you downloaded (and extracted) the dataset
DATASET_PATH = ''
```
5. **run the code**:
```bash
python detection.py
```
the output file is ``` './instance_detection_results' ```