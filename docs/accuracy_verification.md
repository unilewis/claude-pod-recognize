# OCR Accuracy Verification Guide

To verify an accuracy rate of **>95%**, you must perform benchmarking against a "Ground Truth" dataset. Accuracy in OCR isn't just about reading letters; it's about correctly extracting the structured address.

## 1. Preparing the Ground Truth (GT)

You need a representative sample of delivery photos (e.g., 500-1000 images) manually labeled with the correct data.

**Create a `ground_truth.json` file:**

```json
{
  "pod1.jpg": {
    "street_number": "123",
    "street_name": "MAIN ST",
    "unit_number": null
  },
  "pod2.jpg": {
    "street_number": "45",
    "street_name": "ORCHARD CLOSE",
    "unit_number": "Apt 4"
  },
  "pod3.jpeg": {
    "street_number": "8",
    "street_name": "WESTMINSTER RD",
    "unit_number": null
  }
}
```

## 2. Defining "Accuracy"

For POD processing, accuracy is typically measured as **Field-Level Accuracy**:

- **Strict Accuracy**: All fields (Number, Name, Unit) must match exactly.
- **Relaxed Accuracy**: Number and Name must match. Unit is optional (often missing in photos).
- **Character Error Rate (CER)**: Percentage of incorrect characters (useful for fine-tuning but less useful for business logic).

**Formula**:
$$\text{Accuracy} = \frac{\text{Correctly Parsed Images}}{\text{Total Images}} \times 100$$

## 3. Validation Script Template

I recommend creating a `validate_accuracy.py` script that compares OCR output against your JSON ground truth.

```python
import json
import os
from pod_ocr import extract_numbers

def run_validation(gt_file, image_dir):
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)

    total = len(ground_truth)
    correct = 0
    failures = []

    for filename, expected in ground_truth.items():
        path = os.path.join(image_dir, filename)
        result = extract_numbers(path)

        # Comparison logic (Strict)
        match = (
            str(result['street_number']) == str(expected['street_number']) and
            str(result['street_name']).upper() == str(expected['street_name']).upper()
        )

        if match:
            correct += 1
        else:
            failures.append({
                "file": filename,
                "expected": expected,
                "actual": result
            })

    accuracy = (correct / total) * 100
    print(f"Final Accuracy: {accuracy:.2f}%")
    return failures
```

## 4. How to Improve if Accuracy is <95%

If your accuracy is below the target, analyze the `failures` list to identify patterns:

1.  **Preprocessing Issues**: Are images too dark? (Update contrast logic).
2.  **Regex Issues**: Is a specific street suffix being missed? (Update `parse_street_name`).
3.  **Model Issues**: Is the model misreading '8' as 'B'? (Requires fine-tuning PaddleOCR on local font/handwriting styles).
4.  **Confidence Threshold**: If accuracy is low but confidence is high, the model is "confidently wrong". If both are low, the image quality is likely the issue.
