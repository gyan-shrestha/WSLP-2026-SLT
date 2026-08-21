# Pose to Text Translation for Indian Sign Language

Code for our submission to the WSLP 2026 shared task, Task 1 (Sign Language
Translation). The system maps Indian Sign Language pose keypoint sequences to
English sentences using a T5 encoder decoder with an auxiliary contrastive
objective and ensemble decoding.

Reported test set scores: chrF 0.16, ROUGE 0.08, BLEU 0.00.

## 1. Requirements

A machine with one NVIDIA GPU. Training one model takes about one hour on an
NVIDIA B200 and needs roughly 20 GB of GPU memory at the default batch size.

About 200 GB of disk is needed if you download the full pose corpus.

## 2. Environment setup

The project uses conda. Create the environment:

    conda create -y -p ./env python=3.11
    conda activate ./env

Install PyTorch. Use the CUDA build that matches your driver. For CUDA 12.8:

    pip install torch --index-url https://download.pytorch.org/whl/cu128

Install the remaining dependencies:

    pip install "transformers>=4.44" "accelerate>=0.34" sentencepiece
    pip install pose-format numpy pandas tqdm
    pip install sacrebleu rouge nltk
    pip install av opencv-python-headless pillow torchvision

The last line is only needed if you want to extract video features.

Download the NLTK tokenizer data used by the scorer:

    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

Verify the install:

    python -c "import torch, transformers, pose_format, sacrebleu; print(torch.__version__, torch.cuda.is_available())"

## 3. Set the data root

Every script reads one environment variable that points at your data and output
directory. Set it before running anything:

    export SLT_ROOT=/path/to/your/workspace

The scripts expect and create this layout under SLT_ROOT:

    isign/        pose archive and annotations, downloaded in step 4
    task/         shared task validation and test data
    feats/        preprocessed float16 features, created in step 5
    vfeats/       video features, optional
    runs/         training checkpoints and logs
    preds/        generated predictions
    logs/         job output

## 4. Download the data

The corpus is gated on Hugging Face. Request access to
Exploration-Lab/iSign and Exploration-Lab/WSLP first, then authenticate:

    export HF_TOKEN=your_token_here

Download. The pose archive is about 160 GB, so this takes a while:

    bash slurm/download.sbatch

Or run the same commands directly:

    hf download Exploration-Lab/WSLP --repo-type dataset \
        --include "Shared_task_MT/*" --local-dir $SLT_ROOT/task
    hf download Exploration-Lab/iSign --repo-type dataset \
        --include "*.csv" --local-dir $SLT_ROOT/isign
    hf download Exploration-Lab/iSign --repo-type dataset \
        --include "iSign-poses_v1.1_part_*" --local-dir $SLT_ROOT/isign

Note that the pose archive ships as four split parts. The code reads them
without concatenating, so do not join them.

## 5. Preprocess

This decodes the pose files, selects 103 keypoints, normalises them, and writes
packed float16 shards. It is CPU bound and parallel across shards.

    python scripts/preprocess.py --source isign --shard 0 --num-shards 64
    python scripts/preprocess.py --source val   --shard 0 --num-shards 4
    python scripts/preprocess.py --source test  --shard 0 --num-shards 4

Run all shard indices, either in a loop or as a job array. On a cluster:

    sbatch --array=0-63 --export=ALL,SOURCE=isign slurm/preprocess.sbatch

Output goes to $SLT_ROOT/feats. Expect about 11 GB for the full corpus.

## 6. Train

    python src/train/train.py --config configs/norm.json

The configs correspond to results reported in the paper:

    base.json           cross entropy only, no contrastive objective
    contrastive.json    adds the contrastive objective
    norm.json           adds clip level normalisation, our main system
    norm_s1..s4.json    the same as norm.json with different random seeds
    abl_velocity.json   adds velocity features
    abl_augment.json    adds augmentation
    abl_large.json      uses flan-t5-large
    video.json          uses SigLIP video features instead of keypoints

Checkpoints are written to the out_dir named in each config. Each run also
writes log.jsonl with per epoch development scores.

To train the ensemble members, run norm.json and norm_s1 through norm_s4.

## 7. Predict

Single model:

    python src/eval/predict.py \
        --ckpt $SLT_ROOT/runs/norm/best.pt \
        --source test \
        --num-beams 8 --length-penalty 2.0 --min-new-tokens 36 \
        -o $SLT_ROOT/preds/test.csv

Ensemble over several checkpoints, averaging log probabilities at each decoding
step:

    python src/eval/ensemble.py \
        --ckpts $SLT_ROOT/runs/norm/best.pt \
                $SLT_ROOT/runs/norm_s1/best.pt \
                $SLT_ROOT/runs/norm_s2/best.pt \
        --source test --num-beams 8 --length-penalty 2.0 \
        -o $SLT_ROOT/preds/ensemble.csv

To measure an ensemble before spending a submission, use the development split,
which has references:

    python src/eval/ensemble.py --ckpts ... --source dev --limit 800

## 8. Build a submission

The shared task expects a zip containing exactly one file named answer.csv, at
the archive root, with rows in the same order as test.csv.

    python scripts/make_submission.py \
        --preds $SLT_ROOT/preds/ensemble.csv \
        -o submission.zip

The script refuses to build an archive that is missing any uid, and asserts that
the archive contains only answer.csv.

## 9. Optional: video features

The system in the paper uses pose keypoints. We also tested SigLIP embeddings of
sampled video frames, which did not improve on keypoints.

    python scripts/extract_video_feats.py --source test --shard 0 --num-shards 1
    python src/train/train.py --config configs/video.json

Extraction runs at about 6 clips per second on one GPU and is bound by video
decoding rather than by the encoder.

## 10. Reproducing the reported numbers

    Table with the ablation      configs base, contrastive, norm, abl_*
    Table with seed variance     configs norm, norm_s1, norm_s2
    Decoding length results      src/eval/predict.py with --min-new-tokens
    Degenerate baselines         scripts/baselines.py

The metric implementation in src/eval/metrics.py mirrors the shared task scorer:
unsmoothed NLTK BLEU with the punkt tokenizer, the rouge package version 1.0.1,
and sacrebleu chrF.

## Repository layout

    src/data/pose.py        keypoint selection and normalisation
    src/data/dataset.py     memory mapped shard reader, augmentation
    src/data/multipart.py   seekable reader over a split zip archive
    src/models/pose_t5.py   the model, contrastive objective, generation
    src/train/train.py      training loop
    src/train/pretrain.py   masked pose pretraining, an ablation
    src/eval/predict.py     single model inference
    src/eval/ensemble.py    ensemble decoding
    src/eval/sweep.py       decoding parameter sweep
    src/eval/retrieve.py    nearest neighbour retrieval baseline
    src/eval/metrics.py     scorer matching the shared task
    scripts/preprocess.py   pose archive to packed features
    scripts/baselines.py    degenerate baselines used in the analysis

## Notes

Two implementation details caused us significant trouble and are documented in
the code.

T5 does not scale its embeddings by the square root of the model dimension, so
its embedding table has a per token norm near 300 while a LayerNorm output has
norm near 27.7. Passing pose vectors at unit scale makes training begin at a
loss near 609 instead of near 10. See emb_scale in src/models/pose_t5.py.

Normalising each frame by its own shoulder width is unstable when the shoulder
detection is poor, and produced coordinates scaled by up to 500 times. We use a
clip level median instead. See normalize in src/data/pose.py.

## Data license

The iSign corpus is released under its own terms and is not redistributed here.
Obtain it from Hugging Face and cite the benchmark paper as required by its
license.
