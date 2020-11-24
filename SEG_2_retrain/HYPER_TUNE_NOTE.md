# Hyper Tuning versions

As kerastuner is unable to add layers dynamically (if do so, occurs exception on graph_create()), split each version with different number of layers in Encoder/Decoder layers.

1. {SEG_2_retrain_hyper_tuning_1}
    1. Num of Encoder Layers : 1
    2. Num of Decoder Layers : 1
2. {SEG_2_retrain_hyper_tuning_2}
    1. Num of Encoder Layers : 2
    2. Num of Decoder Layers : 1
3. {SEG_2_retrain_hyper_tuning_3}
    1. Num of Encoder Layers : 1
    2. Num of Decoder Layers : 2
4. {SEG_2_retrain_hyper_tuning_4}
    1. Num of Encoder Layers : 2
    2. Num of Decoder Layers : 2