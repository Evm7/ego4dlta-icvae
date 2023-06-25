## 📝 QuickStart
### Install dependencies 
```bash
conda create -n icvae python=3.8
source activate icvae
cd [Path_To_This_Code]
pip install -r requirements.txt
mkdir outputs
```

### Ego4D videos and metadata
- Follow the procedure indicated in [EGO4D main page](https://ego4d-data.org/docs/data/features) on how to download the pre-extracted visual features from Slowfast 8x8 R101 pretrained in Kinetics400:
```bash
 python3 -m ego4d.cli.cli --output_directory="PATH_DO_DIR" --datasets slowfast8x8_r101_k400 --benchmarks FHO
```
- Download all the annotations for FHO_LTA challenge (train,test, and validation) and add them into the annotations directory (together with json files preprocessed by us)/
- Modify config.yaml file and paths.py from data directory by adding the path information to this code ([PATH_TO_CODE_BASE]) and to the feature dataset downloaded from Ego4d and padded ([PATH_TO_DATASET])
- Pre-process the features dataset by using the preprocess_dataset.py script. Adapt the default paths to your enviroment.

## 🏋️‍️ Usage
Our framework is based on two different modules: 
- Hierarchical Multitask MLP Mixer (H3M)
- Intention-Conditioned Variational Autoencoder (I-CVAE)

### Train
Each of the modules is trained independently. To train each module, you only need to modify the $totrain variable inside the train_model.py
When a module is trained, a directory with the resulting module, events and results will be saved in the ligthning_logs.

### Evaluate
Each of the modules can be tested (using the ground-truth verb-noun pairs in the case of I-CVAE) using the test_model.py file. 
It is important to define (in the same script) the version file that will be used for testing. The code reads the cfg file from that directory and loads the adequate model.

### Test
Consists on the sequential use of the H3M module to classify the video features to the action classes, and then use this output as input to the ICVAE module. This is the code needed to obtain the results from our challenge.
Please adapt the versions setted in the file to obtain the best results.

### Configuration
Configuration to obtain the best results can be obtained from the hparams. You need to define the hparams as in the files for the modules used.
Several attributes should be changed from the files to adapt to your environment:
(CHECKPOINT_FILE_PATH, CONFIG_FILE, FEAT_PREFIX, FOLDER, PATH_PREFIX, annotation_path, OUTPUTS_PATH)
Configuration files:
- hparams_h3m.yaml
- hparams_icvae.yaml
- hparams_intention.yaml
