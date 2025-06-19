# Self-attentive Bidirectional LSTM Networks for Temporal Decoding of EEG Motor States

**Authors:** Sara Kamali • Fabiano Baroni • Pablo Varona  

Using any code or material in this repository **requires prior permission from the authors**.  
If you reference the methods or results, please cite the paper as listed above.  
MATLAB and Python source files are provided.

---

## 1. Dataset

* **Task:** stereotypical finger-pinching  
* **Public dataset DOI:** <https://doi.org/10.1093/gigascience/gix034>  
* **Context:** Fig. 1 of the paper shows the task timeline and execution steps.

---

## 2. Pre-processing (MATLAB)

| File | Purpose |
| ---- | ------- |
| `MS_cls_info.mat` | Subject / component indices |
| `initialize_environment.m` | Parameter settings |
| `preparing_data_format_and_labels.m` | Re-formats data and labels for Python |

Detailed EEG pre-processing procedures are available in the companion repository:  
<https://github.com/GNB-UAM/Kamali_Mu_and_beta_power_precue_effects_double_dissociate_latency>

---

## 3. Deep-learning model (Python)

* **Script:** `BiLSTM_Self_attention_EEG_classifier.py`  
  * Implements a BiLSTM followed by a self-attention layer  
  * Generates all evaluation metrics and plots reported in the paper

---


## 4. Additional material

| File | Description |
|------|-------------|
| `IWANN2025_Presentation_Kamali_etal.pdf` | Slide deck used for the IWANN 2025 conference presentation |

---

## 5. Reference

> Kamali S., Baroni F., & Varona P. (2025).  IWANN Conference.
> *Self-attentive Bidirectional LSTM Networks for Temporal Decoding of EEG Motor States.*

