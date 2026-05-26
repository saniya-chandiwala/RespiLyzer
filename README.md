# RespiLyzer

AI-based system for Respiratory Disease Classification from Lung Sounds

RespiLyzer evaluates respiratory audio data to detect lung abnormalities and classify specific respiratory diseases. Utilizing advanced signal processing techniques alongside an ensemble of Convolutional Neural Networks (CNNs), it analyzes acoustic patterns from lung sounds to provide accurate insights that assist clinicians in early diagnostics and treatment planning.

## Dataset

The system is developed using the official [**ICBHI 2017 Respiratory Sound Database**](https://bhichallenge.med.auth.gr).

* It contains 920 annotated audio recordings from 126 unique patients.
* The audio inputs are standardized into a uniform 6-second window before extraction to maintain structural input consistency during training.
* Which are further classified across clinical conditions including Asthma, COPD, Pneumonia, and normal respiratory cycles.

## Feature Extraction Techniques

To capture the complex frequency patterns of lung anomalies, the pipeline extracts three primary audio features using librosa:
1. **MFCC (Mel-Frequency Cepstral Coefficients):** Captures the power spectrum of the audio to model the overall timbre of breathing sounds.
2. **Chroma STFT (Short-Time Fourier Transform):** Maps the audio energy onto the distinct chromatic musical pitches to capture tonal variations.
3. **mSpec (Mel Spectrogram):** Converts frequencies to the non-linear Mel scale to closely mimic human auditory perception of sound anomalies.

## How It Works

* Upload .wav files of respiratory sounds to predict underlying lung conditions.
* Ensemble Learning Framework integrates deep learning models trained on distinct audio feature extraction methods for robust evaluation.
* Then Granular Model displays individual predictions from each feature extraction approach alongside a final primary diagnosis classification.

## Tech Stack

* **Frontend & Interface:** Streamlit
* **Backend Engine:** Python
* **Signal Processing:** Librosa, SoundFile
* **Deep Learning Framework:** PyTorch / TensorFlow Keras

## Models Used

Separate deep learning architectures were trained on each isolated audio feature mapping to create the ensemble structure:
* **MFCC Model:** Specialized CNN trained on cepstral coefficients.
* **Chroma Model:** Optimized network processing Short-Time Fourier structural pitches.
* **Mel Spectrogram Model:** CNN evaluating log-scaled frequency spectrogram matrices.

## Screenshots

![Dashboard View](./imgs/Application%20Dashboard.png)
![Dashboard View](./imgs/Interface%20Details.png)
![Dashboard View](./imgs/Predicted%20Output.png)
---

## How to Run Locally

1. Clone the repository:

   `git clone [https://github.com/saniya-chandiwala/RespiLyzer]`
   
   `cd RespiLyzer`
      
3. Run the application:
  `streamlit run app.py`
