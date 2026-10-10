# Paper selection (15 of the 45 supplied summaries)

Source: the three reading-summary tables (Audio, Multimodal, Video). Identifiers are the table letter plus the paper number (A = audio, M = multimodal, V = video).

Authors are not part of the summary tables. Where the original methodology DOCX reference list gives authors for the same paper, those authors are used; otherwise the field says 'not in summaries'. No bibliographic detail was added from memory.

Data-quality flags found in the supplied summaries:

- V9 and V14 are the same paper (GenConViT) listed twice, so the set holds 44 distinct papers.
- M2 has a garbled title in the table row ('Media Deepfake Detection Beyond Face Swapping: ...'); the title in the original DOCX reference list ('Beyond face swapping: A diffusion-based digital human benchmark for multimodal deepfake detection') was used.
- M1 (Omni-Fake) is dated 2025 in the table and CVPR 2026 in the original DOCX; it is excluded, so no change was needed.
- A9 (mixture-of-experts, frozen wav2vec 2.0) is arXiv:2409.11909 (2024) in the table, while the original DOCX listed it as ICASSP 2025. The table was followed; please verify the venue.
- A1: the Limitations cell repeats the Technologies cell; limitation statements for A1 were therefore not used.
- None of the 45 summaries describes AV-Deepfake1M++ itself or LAV-DF's own dataset paper; M7 and M8 are the only papers that use LAV-DF, and M8 uses AV-Deepfake1M. The dataset descriptions of LAV-DF and AV-Deepfake1M++ in the revised document are flagged for verification.
- No summary covers Whisper or Sentence-BERT, so those components are described as design choices without a citation.

## Selected (15)

| # | ID | Title | Authors | Year | Main contribution | Relevance | Reason |
|---|----|-------|---------|------|-------------------|-----------|--------|
| 1 | A1 | A Superb-Style Benchmark Of Self- Supervised Speech Models For Audio Deepfake Detection | H. Ali, N. S. Adupa, S. Subramani, H. Malik | 2026 | SUPERB-style benchmark of 11 SSL models | Compares wav2vec 2.0 with other SSL encoders; includes Deepfake-Eval-2024 | INCLUDE - audio benchmark methodology |
| 2 | A9 | Mixture of Experts Fusion for Fake Audio Detection Using Frozen wav2vec 2.0 | Z. Wang et al. (13 authors) | 2024 | Mixture-of-experts fusion of frozen wav2vec 2.0 layers | Frozen wav2vec 2.0 for audio deepfakes | INCLUDE - audio encoder justification |
| 3 | A14 | Comprehensive Layer-wise Analysis of SSL Models for Audio Deepfake Detection | Y. El Kheir, Y. Samih, S. Maharjan, T. Polzehl, S. Moller | 2025 | Layer-wise analysis of SSL models for audio deepfake detection | Frozen SSL encoders, layer aggregation; limits of last-layer pooling | INCLUDE - audio refinement / limitation |
| 4 | M2 | Beyond face swapping: A diffusion-based digital human benchmark for multimodal deepfake detection | J. Liu, J. Wang, S. Hou, M. Ren, H. Wu, L. Ma, R. Pei, Z. He | 2026 | DigiFakeAV diffusion-based digital-human benchmark (60,000 videos) | Modern generators, cross/self-attention fusion detector | INCLUDE - realistic-generator robustness |
| 5 | M3 | Divide and Conquer: Multimodal Video Deepfake Detection via Cross-Modal Fusion and Localization | Q. Li, M. He, L. Yi, Q. Wen, Y. Zhang, H. Jin, P. Cheng, Z. Ba, L. Lu, K. Ren | 2026 | Modular multimodal detection and localisation (challenge system) | Contrast: independently trained modules, no shared features | INCLUDE - motivates joint fusion |
| 6 | M5 | Intelligent Deepfake Detector Using Audio-Visual Clues | B. G. Banik, S. N. Naziya | 2026 | Dual-stream transformer, MTCNN, wav2vec 2.0 | Closest pipeline match (MTCNN + wav2vec 2.0); future work asks for text transcripts | INCLUDE - supports tri-modal motivation |
| 7 | M6 | DF-TransFusion:Multimodal Deepfake Detection via Lip- Audio Cross-Attention and Facial Self-Attention | A. Kharel, M. Paranjape, A. Bera | 2023 | Lip-audio cross-attention + facial self-attention | Attention-based audio-visual fusion; lip-sync cue; names missing semantic analysis | INCLUDE - closest fusion design reference |
| 8 | M7 | UMMAFormer: A Universal Multimodal-adaptive Transformer Framework for Temporal Forgery Localization | R. Zhang, H. Wang, M. Du, H. Liu, Y. Zhou, Q. Zeng | 2023 | Multimodal-adaptive transformer for temporal forgery localisation | LAV-DF benchmark paper; temporal evidence | INCLUDE - LAV-DF coverage |
| 9 | M8 | Contextual Cross-Modal Attention for Audio-Visual Deepfake Detection and Localization | V. S. Katamneni, A. Rattani | 2024 | Contextual cross-modal attention, detection + localisation | Covers FakeAVCeleb, AV-Deepfake1M, LAV-DF; names semantic text stream as future work | INCLUDE - links two selected datasets and the AV-Deepfake1M family |
| 10 | M9 | AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection | T. Oorloff et al. (9 authors) | 2024 | Self-supervised audio-visual fusion (ViT, contrastive, complementary masking) | Core audio-visual fusion baseline on FakeAVCeleb; names missing semantic layer | INCLUDE - fusion design reference, selected-dataset benchmark |
| 11 | M10 | FakeAVCeleb: A Novel Audio- Video Multimodal Deepfake Dataset | H. Khalid, S. Tariq, M. Kim, S. S. Woo | 2021 | FakeAVCeleb dataset paper | Selected dataset; includes EfficientNet-B0 baseline | INCLUDE - dataset reference |
| 12 | M14 | SAVe: Self-Supervised Audio-visual Deepfake Detection | S. A. Shahzad et al. (8 authors) | 2026 | Self-supervised AV detection with self-blended pseudo-artifacts | Visual artifacts + misalignment; generator-bias limitation | INCLUDE - generalisation / generator bias |
| 13 | M15 | Audio-Visual Deepfake Detection With Local Temporal Inconsistencies | M. Astrid, E. Ghorbel, D. Aouada | 2025 | Local temporal inconsistencies, temporal distance map | Explicit temporal AV evidence on FakeAVCeleb | INCLUDE - supports separate lip-sync cue |
| 14 | V4 | Cross-Dataset Video Deepfake Detection Using Transformer and CNN Architectures | G. Petmezas | 2026 | Cross-dataset study of transformer and CNN video detectors | Generalisation evaluation methodology | INCLUDE - evaluation / generalisation |
| 15 | V8 | Exploring Temporal Coherence for More General Video Face Forgery Detection (FTCN) | Y. Zheng, J. Bao, D. Chen, M. Zeng, F. Wen | 2021 | FTCN: fully temporal convolution + temporal transformer | Temporal coherence; future work asks for audio and semantic layers | INCLUDE - temporal-cue justification |

## Excluded (30)

| ID | Title | Year | Reason for exclusion |
|----|-------|------|----------------------|
| A2 | DeepRawNet: Empowering Deepfake Audio Detection through Dynamic Enhancements | 2026 | Raw-waveform audio-only architecture; no multimodal or frozen-SSL link |
| A3 | Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception | 2026 | All-type audio (music, sound, singing); outside scope |
| A4 | ESDD 2026: Environmental Sound Deepfake Detection Challenge Evaluation Plan | 2026 | Environmental sound challenge; outside speech scope |
| A5 | PhenomeDF: A Synthetic Speech Dataset for Audio Deepfake Detection and Naturalness Evaluation | 2026 | Phoneme-level speech dataset; not a selected dataset |
| A6 | Robust DeepFake Audio Detection via an Improved NeXt- TDNN with Multi- Fused Self- Supervised Learning Features | 2025 | Audio-only NeXt-TDNN with fused SSL features; overlaps with kept SSL papers |
| A7 | Deepfake Audio Detection with Spectral Features and ResNeXt-based Architecture | 2025 | Spectral ResNeXt audio-only; hand-crafted feature line |
| A8 | RawBMamba: End-to- End Bidirectional State Space Model for Audio Deepfake Detection | 2024 | Mamba end-to-end audio model; architecture not used here |
| A10 | Audio Deepfake Detection in the Age of Advanced Text-to- Speech Models | 2026 | TTS-era audio detection; overlaps A9/A14/A1 (also uses Whisper-style features); lower priority |
| A11 | Deep Fake Audio Detection Using a Hybrid CNN-BiLSTM Model with Attention Mechanism | 2025 | CNN-BiLSTM audio, low-tier venue, SceneFake/FoR data |
| A12 | TRACE: Training-Free Partial Audio Deepfake Detection via Embedding Trajectory Analysis of Speech Foundation Models | 2026 | Partial-spoof training-free method; fails on fully synthesized speech (limitation in summary) |
| A13 | XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection | 2026 | XLSR-MamBo audio backbone; architecture not used here |
| A15 | Quantum-Trained Convolutional Neural Network for Deepfake Audio Detection | 2024 | Quantum-trained CNN; unrelated to our design |
| M1 | Omni-Fake: Benchmarking Unified Multimodal Social | 2025 | 7B unified multimodal LLM detector, social-media scope; year differs between summary (2025) and original DOCX (2026); heavy and less aligned |
| M4 | TalkingHeadBench: A Multi- Modal Benchmark & Analysis of Talking-Head DeepFake Detection | 2025 | Talking-head benchmark without a detection model; DigiFakeAV [M2] covers modern generators more directly |
| M11 | Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes | 2026 | In-the-wild evaluation benchmark; authors not given in the summaries or the original DOCX |
| M12 | Multi-Modal Solution: Deepfake Detection and the Source Identification | 2024 | Competition write-up reporting a gradient explosion; low methodological value |
| M13 | LEVERAGING LARGE MULTIMODAL MODELS FOR AUDIO-VIDEO DEEPFAKE DETECTION: A PILOT STUDY | 2026 | Large-multimodal-model pilot; needs task-specific tuning, not our architecture |
| V1 | An Optimal Deep Learning-Based Face Forgery Detection System Using DCXCN- BLSTM | 2026 | Video-only forgery detector on SiW-M/DFFD/CASIA |
| V2 | Robust Deepfake Video Detection Using Spatio-Temporal Features and Dynamic Difference Learning | 2026 | Video-only spatio-temporal model on FF++/DFDC/UADFV (not selected datasets) |
| V3 | Spatiotemporal Deep Learning for Real-Time Video-Based Deepfake Detection Using 3DCNN, 3DResNet, TCN, and VAE | 2026 | Video-only 3D-CNN/TCN/VAE on FF++/DFDC/Celeb-DF |
| V5 | A Gated Temporal Attention Based Intra Prediction Framework for Robust Deepfake Video Detection | 2025 | Video-only, single dataset (Celeb-DF v2) |
| V6 | FaceForensics++: Learning to Detect Manipulated Facial Images | 2019 | FF++ dataset paper; FF++ is not a selected dataset (kept as related-work mention only) |
| V7 | Celeb-DF: A Large- Scale Challenging Dataset for DeepFake Forensics | 2020 | Celeb-DF dataset paper; Celeb-DF is not a selected dataset |
| V9 | GenConViT: Deepfake Video Detection Using Generative Convolutional Vision Transformer | 2023 | GenConViT (video-only; duplicate of V14 in the supplied set) |
| V10 | Deepfake Defense: Combining Spatial and Temporal Cues with CNN-BiLSTM- Transformer Architecture | 2024 | Video-only CNN-BiLSTM-Transformer on FF++/DFDC |
| V11 | MesoNet: A Compact Facial Video Forgery Detection Network | 2018 | MesoNet: early compact CNN; superseded for our purposes by V4/V8 |
| V12 | Explainable Deepfake Video Detection using Convolutional Neural Network and CapsuleNet | 2024 | Video-only CapsuleNet on DFDC |
| V13 | Deepfake Video Detection Using Convolutional Vision Transformer (CViT) | 2021 | CViT: video-only, DFDC; not needed beyond V4/V8 |
| V14 | GenConViT: Deepfake Video Detection Using Generative Convolutional Vision Transformer | 2025 | Duplicate title of V9 in the supplied set; video-only |
| V15 | Combating Deepfakes: A Comprehensive Multilayer Deepfake Video Detection Framework | 2024 | Video-only multilayer framework on FF++/DFD/Celeb-DF |
