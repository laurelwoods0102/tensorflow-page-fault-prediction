# Tensorflow Page Fault Prediction

## Dataset
1. GEMM ({GEMM_21gb_gpa_pf_stride.csv})
2. GEMM_EX ({gem_3214_generic_*.csv})
3. NU
4. SEG
5. StreamBench
    1. 1G1P
    2. 2G1P
6. Stream_14gb_2vm

***

## Preprocessing
1. {SEG_preprocess_pipeline} : First pipeline (NOTE : Unable to conduct Inverse_transform properly)
2. {SEG_preprocess_pipleline_retrain} : Second pipeline for {retrain} models.

***

## Model Architectures/Versions
1. SEG Dataset
    1. SEG : Encoder-Decoder 
        1. {SEG} : Dataset Pruning Threshold = 50
        2. {SEG_2} : Dataset Pruning Threshold = 2 / HParms Tuned
        3. {SEG_retrain} : Retrained SEG_2
    2. SEG_AR : Autoregressive LSTM
        1. {SEG_AR} : Vanilla Autoregressive LSTM
        2. {SEG_AR_variants} : variants of LSTM cells (stacked LSTM / multiple LSTM / Peepholed LSTM)
    3. SEG_CNNLSTM : CNN + Encoder-Decoder {CNNLSTM_model} / CNN-LSTM {CNNLSTM_model_2} (Mistakingly, didn't separate models)
        1. {SEG_CNNLSTM_1_retrain} : Retrained CNN + Encoder-Decoder 
        2. {SEG_CNNLSTM_2_retrain} : Retrained CNN-LSTM
    4. SEG_Wavenet : Wavenet (Dilated Causal Convolution)
        1. {SEG_Wavenet} : Wavenet model with {SEG_preprocess_pipeline_2} dataset (Same dataset as {retrain} models)

2. NU Dataset
    1. NU : Encoder-Decoder
        1. {NU} : Encoder-Decoder model
    2. NU_AR : Autoregressive LSTM
        1. {NU_AR} : AR-LSTM model
