# Siamese_Based Visual Object Tracking Model

<div style="text-align: justify">
This study presents a novel Siamese architecture for visual tracking show in Figure. Twin backbone networks independently extract feature representations from the template and the current frame. Cross-correlating these representations produces a response map from which the target's spatial coordinates and segmentation mask are inferred.
The backbone leverages a high-efficiency Visual State Space Model (VSSM), synergistically coupling the sequential-reasoning capability of state-space models with the locality-sensitive feature extraction of convolutional neural networks, thereby achieving an optimal trade-off between computational overhead and tracking fidelity.
The response map is subsequently refined by a Kalman-filter-based motion-estimation module, which supplies temporal priors to guide the mask decoder. Finally, down-sampled features from the current frame are archived in a memory bank, enabling their reuse as the template for subsequent frames.
</div>

![Model Architecture](https://raw.githubusercontent.com/Simon9623/SiamEVM/91321ebfa5b9a5e7e953ddf2312f605469894b94/Documents/model.svg)

# Installation

# Usage

