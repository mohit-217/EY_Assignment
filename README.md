# Model Training Logs

## Training Details
This document contains the training logs for the fine-tuning of a custom model on financial news data.

### Training Logs
| Epoch | Loss  | Grad Norm | Learning Rate |
|-------|-------|------------|---------------|
| 0.02  | 4.6665 | 5.8536      | 2.00E-05      |
| 0.02  | 5.5837 | 9.7500      | 1.99E-05      |
| 0.03  | 5.1382 | 9.2048      | 1.96E-05      |
| 0.04  | 5.2510 | 7.1939      | 1.92E-05      |
| 0.05  | 4.9219 | 4.6888      | 1.85E-05      |
| 0.06  | 5.4368 | 7.4570      | 1.78E-05      |
| 0.06  | 5.4997 | 8.3967      | 1.68E-05      |
| 0.07  | 5.2845 | 7.9477      | 1.58E-05      |
| 0.08  | 4.9487 | 5.9581      | 1.46E-05      |
| 0.09  | 5.8315 | 5.7831      | 1.33E-05      |
| 0.09  | 5.5685 | 7.0311      | 1.20E-05      |
| 0.10  | 4.6686 | 6.0247      | 1.07E-05      |
| 0.11  | 4.9365 | 6.7111      | 9.32E-06      |
| 0.12  | 5.3642 | 6.9885      | 7.97E-06      |
| 0.13  | 5.2558 | 7.7092      | 6.65E-06      |
| 0.13  | 5.0214 | 5.5544      | 5.40E-06      |
| 0.14  | 5.1323 | 7.0828      | 4.23E-06      |
| 0.15  | 4.7463 | 5.4712      | 3.17E-06      |
| 0.16  | 5.7296 | 6.1974      | 2.24E-06      |
| 0.17  | 5.4779 | 5.2142      | 1.46E-06      |
| 0.17  | 5.4575 | 4.5068      | 8.28E-07      |
| 0.18  | 4.8112 | 7.5427      | 3.71E-07      |
| 0.19  | 5.2767 | 7.0348      | 9.31E-08      |
| 0.20  | 4.7541 | 5.2969      | 0.00E+00      |

---

## Model Training Instructions
### To Train the Model
Uncomment the following command and run it:
```bash
#!accelerate launch -m axolotl.cli.train /content/test_axolotl.yaml
```

### To Launch Gradio Demo
Run the following command:
```bash
!accelerate launch -m axolotl.cli.inference /content/test_axolotl.yaml \ 
    --lora_model_dir="mohit9999/all_news_finance_sm_1h2023_custom_model" --gradio
```
You can try multiple models by changing the model name find the below list of models.
---

## Dataset and Model Information
- **Dataset URL:** [Hugging Face Dataset](https://huggingface.co/mohit9999/all_news_finance_sm_1h2023_custom)
- **Model1 and Logs URL:** [Hugging Face Model](https://huggingface.co/mohit9999/all_news_finance_sm_1h2023_custom_model)
- **Model2 and Logs URL:** [Hugging Face Model](https://huggingface.co/mohit9999/all_news_finance_sm_1h2023_custom_model_2)
- **Model2 and Logs URL:** [Hugging Face Model](https://huggingface.co/mohit9999/all_news_finance_sm_1h2023_custom_model_3)

---

## Challenges Faced
- `flash-infer` was **not compatible** with T4 GPUs, causing issues during training.

---

## Dataset Preprocessing
The dataset has been preprocessed according to the **Alpaca format**. The preprocessing script is available in the repository.

---

### Notes
- Ensure `accelerate` is installed and configured properly before running the training or inference commands.
- Use `--gradio` flag to deploy a Gradio demo for easy interaction with the fine-tuned model.

---
