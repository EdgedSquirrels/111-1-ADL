# Homework 1 ADL NTU

## Set Up Environment
```shell
# If you have conda, it's recommended to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip instsall -r requirements.in
```

## Download files
To get the model, it is necessary to get it from Dropbox with download.sh
```shell
bash download.sh
```

## Reproduce the data
Since the downloaded files include the saved models, you may reproduce the result directly.
```
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
```

## Training
### Intent detection
```shell
python train_intent.py
```
The trained model will be saved as `ckpt/intent/model.pt` by default.
For more argument information, you may run `python train_intent.py -h`.

### Slot tagging
```shell
python train_slot.py
```
The trained model will be saved as `ckpt/slot/model.pt` by default. For more argument information, you may run `python train_slot.py -h`.
