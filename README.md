# OpenAI Hackathon Agent

## Overview
This project consists of two main components:

- **flet-app**: A user interface built with Flet for interacting with the agent.
- **finetune**: Scripts for distilling the openai-oss-20B model into the Qwen3 1.7B model using qLoRA and knowledge distillation techniques.

These components work together to enable efficient model training, evaluation, and user interaction.

---

## flet-app

### Purpose
Provides a graphical interface for users to interact with the agent, submit prompts, and view results.

### Main Files
- `main.py`: Entry point for the Flet application. Handles UI logic and communication with backend scripts.

> **Note:** The `build/` directory and `main.spec` file are not included in the GitHub repository. You will need to generate them locally if you wish to build the app as an executable.

### How it Works
1. Launches a Flet-based UI for user interaction.
2. Sends user input to backend scripts (from the `finetune` folder) for processing.
3. Displays results and feedback in the UI.

---

## finetune

### Purpose
Contains scripts for distilling the openai-oss-20B model into the Qwen3 1.7B model, as well as for evaluation and data labeling.

### Main Files
- `distill_qLoRA_eta.py`: Fine-tunes a language model using qLoRA and knowledge distillation.
- `eval.py`: Evaluates the performance of a fine-tuned model.
- `label_with_teacher_multiapi.py`: Uses a teacher model (or multiple APIs) to label data for supervised fine-tuning.
- `prompt_builder.py`: Utilities for constructing prompts for training and evaluation.

### How it Works
- **Distillation & Fine-tuning**: Use `distill_qLoRA_eta.py` to train the Qwen3 1.7B student model from the openai-oss-20B teacher model with efficient memory usage and teacher guidance.
- **Evaluation**: Use `eval.py` to assess model accuracy and performance on test datasets.
- **Labeling**: Use `label_with_teacher_multiapi.py` to generate high-quality labeled data using teacher models or APIs.
- **Prompt Building**: Use `prompt_builder.py` to standardize and generate prompts for various tasks.

---

## Workflow
1. **Label Data**: Run `label_with_teacher_multiapi.py` to create labeled datasets.
2. **Build Prompts**: Use `prompt_builder.py` to format data for training.
3. **Fine-tune Model**: Train a model with `distill_qLoRA_eta.py` using the labeled and formatted data.
4. **Evaluate Model**: Assess the trained model with `eval.py`.
5. **Interact via UI**: Use the Flet app (`main.py`) to interact with the agent and view results.

---

## Requirements
- Python 3.8+
- Flet
- PyInstaller (for building the app)
- PyTorch, Transformers, and other ML dependencies (see individual script requirements)

---

## Usage
1. Install dependencies: `pip install -r requirements.txt` (create this file as needed).
2. Run the Flet app: `python main.py` from the `flet-app` folder.
3. Use the scripts in `finetune` for training and evaluation as described above.

---

## Notes
- Ensure all paths and environment variables are set correctly for model files and datasets.
- For large-scale training, use appropriate hardware (GPUs).
- Refer to comments in each script for detailed usage instructions.
