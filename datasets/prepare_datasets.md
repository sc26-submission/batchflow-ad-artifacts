# Prepare Datasets

## ImageNet-1K

### 1. Download

To access the dataset, first request permission from the official ImageNet website:  
https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php#images  

> **Note:** Downloads are only enabled after your request is approved.

Once approved:

1. Log in to your ImageNet account  
2. Navigate to the ILSVRC 2012 download page  
3. Under **Images**, download the following files:
   - `ILSVRC2012_img_train.tar` — training images  
   - `ILSVRC2012_img_val.tar` — validation images  

```bash
# Download ImageNet images
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

# Download validation labels
wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
```

---

### 2. Process & Upload

Install dependencies and configure AWS:

```bash
pip install boto3
aws configure
```

Run the preprocessing and upload script:

```bash
python imagenet_to_s3.py \
  --train_tar /path/to/ILSVRC2012_img_train.tar \
  --val_tar /path/to/ILSVRC2012_img_val.tar \
  --val_map /path/to/imagenet_2012_validation_synset_labels.txt \
  --s3_bucket <your-bucket-name> \
  --s3_prefix imagenet-1k
```

---

## Open Images

Open Images stores the image files flat rather than in class folders. The W2
loader therefore uses the official image-level annotation CSV to construct the
classification sample index.

### 1. Sync the training images

```bash
aws s3 sync \
  s3://open-images-dataset/train \
  s3://<your-bucket>/open-images/train
```

The resulting layout is:

```text
s3://<your-bucket>/open-images/train/<ImageID>.jpg
```

### 2. Upload the boxable image-level annotations

Download the Open Images boxable human image-label annotations and the
boxable class-description CSV, then upload them once:

```bash
aws s3 cp train-annotations-human-imagelabels-boxable.csv \
  s3://<your-bucket>/open-images/annotations/

aws s3 cp class-descriptions-boxable.csv \
  s3://<your-bucket>/open-images/annotations/
```

Update `batchflow/config/dataset/openimages.yaml` if a different bucket or
prefix is used. On the first run, BatchFlow constructs a dataset manifest from
the positive annotations; later runs reuse the cached manifest.


---
## COCO (W3 retrieval)

W3 uses the ALBEF COCO retrieval training annotations based on the Karpathy
split. The annotation file is `coco_train.json`. Its `image` fields refer to
files below a common COCO image root, including both `train2014/` and
`val2014/`.

### 1. Download COCO 2014 images

```bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

unzip train2014.zip
unzip val2014.zip
```

### 2. Download the ALBEF retrieval annotations

The official ALBEF repository distributes its downstream-task JSON files in
`data.tar.gz`:

```bash
wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/data.tar.gz
tar -xzf data.tar.gz
```

The extracted data include `coco_train.json`, `coco_val.json`, and
`coco_test.json`. W3 training only requires `coco_train.json`.

### 3. Upload the W3 inputs to S3

```bash
aws s3 sync train2014/ s3://<your-bucket>/coco/train2014/
aws s3 sync val2014/ s3://<your-bucket>/coco/val2014/

aws s3 cp data/coco_train.json \
  s3://<your-bucket>/coco/annotations/coco_train.json
```

The expected layout is:

```text
s3://<your-bucket>/coco/
├── train2014/
│   └── ...
├── val2014/
│   └── ...
└── annotations/
    └── coco_train.json
```

Update `batchflow/config/dataset/coco.yaml` with the bucket name. The first
run converts the ALBEF/Karpathy JSON into BatchFlow's canonical dataset
manifest. Each caption is a logical training sample, and all captions for the
same `image_id` share the same contiguous retrieval identifier. All evaluated
systems consume this same manifest.

