# PaddleOCR Fine-tuning Guide

Fine-tuning is necessary if the default models struggle with specific fonts, layouts, or handwriting patterns found in your POD images.

## 1. Data Preparation (Labeling)

You need ~500-2000 images for effective fine-tuning.

- **Tool**: Use **[PPOCRLabel](https://github.com/PaddlePaddle/PaddleOCR/tree/main/PPOCRLabel)** (the official semi-automatic labeling tool).
- **Process**:
  1.  Load images into PPOCRLabel.
  2.  Click "Auto Recognition" to let the model do the first pass.
  3.  Manually correct the bounding boxes and text labels.
  4.  Export the results as `Label.txt`.

## 2. Dataset Formatting

Organize your data into the PaddleOCR format:

```text
/dataset/
  ├── train/
  │   ├── img_1.jpg
  │   └── ...
  ├── val/
  │   ├── img_101.jpg
  │   └── ...
  ├── train_list.txt  (format: img_path \t label_json)
  └── val_list.txt
```

## 3. Choosing a Base Model

For POD recognition, fine-tune the **Recognition (Rec)** model (the one that reads text), not just the Detection (Det) model.

- **Recommended**: `en_PP-OCRv4_mobile_rec` (fast and efficient).

## 4. Training Configuration (YAML)

Modify the configuration file (e.g., `en_PP-OCRv4_mobile_rec.yml`):

- **`Train.dataset.data_dir`**: Path to your training images.
- **`Train.dataset.label_file_list`**: Path to `train_list.txt`.
- **`Optimizer.lr`**: Start with a small learning rate (e.g., `0.001`).
- **`Global.epoch_num`**: Usually 100-200 epochs is enough for fine-tuning.

## 5. Running the Training

Use the PaddleOCR training script:

```bash
python3 tools/train.py -c configs/rec/PP-OCRv4/en_PP-OCRv4_mobile_rec.yml \
    -o Global.pretrained_model=./pretrain_models/en_PP-OCRv4_mobile_rec_train/best_accuracy
```

## 6. Model Export for Production

Once training is complete, convert the "training model" into an "inference model" for use in your `pod_ocr.py` script:

```bash
python3 tools/export_model.py -c configs/rec/PP-OCRv4/en_PP-OCRv4_mobile_rec.yml \
    -o Global.pretrained_model=./output/rec_ppocr_v4/best_accuracy \
    Global.save_inference_dir=./inference/rec_ppocr_v4/
```

## 7. When to Fine-tune?

| Scenario                                    | Action                                 |
| :------------------------------------------ | :------------------------------------- |
| **Blurry Photos**                           | Improve **Image Preprocessing** first. |
| **New Address Format**                      | Improve **Regex Parsing** first.       |
| **Incorrect Characters** (e.g., 'S' as '8') | **Fine-tune** the Recognition model.   |
| **Missing Text Blocks**                     | **Fine-tune** the Detection model.     |
