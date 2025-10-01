# Audio-Aware-Decoding

The core of this method is the logit-processor. You can apply this logit processor to all models and make some changes if needed. For SALMONN and any other audio language model, just make sure you either create all-zero audio or just pass None to the audio argument if that audio language model supports not passing audio.

Paper: https://arxiv.org/pdf/2506.07233
<img width="1860" height="1375" alt="image" src="https://github.com/user-attachments/assets/f7051865-8bba-4957-842e-32ea87e7d84c" />

Dataset: https://github.com/kuan2jiu99/audio-hallucination/tree/main/interspeech2024
