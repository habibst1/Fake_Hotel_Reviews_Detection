# Hotel Review Classification with BERT and Explainability

This repository contains a Jupyter Notebook (`PFA_1.ipynb`) that demonstrates fine-tuning a BERT model for hotel review classification (truthful/deceptive) and explaining the predictions using SHAP and Gemini API.

## Workflow Overview

1. **Data Loading & Exploration**
   - Loaded a dataset containing hotel reviews labeled as "truthful" or "deceptive"
   - Performed exploratory data analysis (EDA) with visualizations
   - Conducted text preprocessing

2. **Model Training**
   - Fine-tuned a BERT model for text classification
   - Evaluated model performance and printed metrics

3. **Model Testing**
   - Tested the trained model on unseen data

4. **Explainability**
   - Used SHAP explainer to identify:
     - Most truthful tokens for each review
     - Most deceptive tokens for each review
   - Created a prompt template containing:
     - User review
     - BERT prediction label
     - Prediction confidence score
     - Important tokens identified by SHAP
   - Fed this prompt to Gemini API to generate human-readable explanations

# Snapshots from the final interface
### Example 1
![image](https://github.com/user-attachments/assets/0dd6ab08-d50a-4ab2-86e3-e2e58f1e97d9)

### Example 2
![image](https://github.com/user-attachments/assets/2a0dce3d-5477-4826-b6b5-9a96d5eea46b)
