# Galapagos Seals Species Classification Project

This repository presents a comprehensive deep learning project focused on the precise identification of two key Galapagos seal species: the **Galapagos Fur Seal** (*Arctocephalus galapagoensis*) and the **Galapagos Sea Lion** (*Zalophus wollebaeki*).

## Making predictions

To use the best-performing InceptionV3 model (from the Hybrid case) to predict on new images:

1. **Open the prediction notebook:** Navigate to Notebooks/Predict-new-case_SealClassification.ipynb.

2. **Mount Google Drive:** Ensure your Google Drive is mounted within the Colab environment to access the saved model.

3. **Verify Model Path:** Confirm that the model_path variable in the notebook correctly points to your saved best_model.h5 (e.g., in Results/inceptionv3_classification_hybrid_case/).

4. **Upload Image:** Run the cell that prompts you to upload a new image file.

5. **View Prediction:** The notebook will then display the predicted species class and its confidence score.

## Project Overview

The core of this project involves developing and evaluating robust image classification models. Key aspects include:
* **Transfer Learning:** Utilizing powerful pre-trained convolutional neural networks (like InceptionV3, ResNet50V2, and VGG16).
* **Class Imbalance Handling:** Implementing hybrid data balancing strategies (undersampling combined with class weights) to address the imbalanced nature of the dataset.
* **Robust Evaluation:** Employing K-fold cross-validation to provide a statistically reliable estimate of model performance across different data splits.

The ultimate goal is to build a reliable AI-powered solution for real-time species identification, significantly aiding conservation efforts and ecological monitoring in the Galapagos Islands.


## Data

The raw image datasets used for training and evaluation are **not included** directly in this repository due to their size. The `combined_dataset_for_kfold` (and its precursor `train`, `test`, `validation` splits) was created by combining and preprocessing images.

To replicate the project or train models, you will need to obtain the original image data. Please refer to the `DataCollection` notebooks for detailed instructions on data gathering and preparation, or contact the repository owner for access to the prepared dataset.

## Setup and Running the Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```
2.  **Ensure Git LFS is installed and configured:**
    ```bash
    # (Run this in your environment/Colab setup)
    apt-get update && apt-get install git-lfs
    git lfs install
    git lfs track "*.h5" # This is already in the .gitattributes if you run this setup
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Data Preparation:** Obtain the necessary image data as described in the 'Data' section and organize it as expected by the notebooks.
5.  **Run Notebooks:** Open the Jupyter notebooks (e.g., in Google Colab) starting from `DataCollection` and proceed through `ObjectClassification` to `k-fold_cross-validation` to understand and replicate the project workflow.

## Key Results (from K-Fold Cross-Validation)

The K-fold cross-validation on the InceptionV3 model with a hybrid balancing strategy yielded the following average performance:

```json
{
    "Average Overall Accuracy": "0.9435 +/- 0.0198",
    "Average Minority Precision": "0.7661 +/- 0.0833",
    "Average Minority Recall": "0.8444 +/- 0.0544",
    "Average Minority F1-score": "0.8018 +/- 0.0635",
    "Average Confusion Matrix (rows are true, columns are predicted)": [
        [7.6, 1.4],
        [2.4, 55.8]
    ]
}
