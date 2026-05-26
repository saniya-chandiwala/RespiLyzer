import io
import numpy as np
import librosa
import streamlit as st
from tensorflow.keras.models import load_model

# --- 1. Page Configuration & Custom CSS ---

st.set_page_config(
    page_title="Respiratory Sound Analysis", 
    layout="wide" 
)

# Custom CSS for dark theme, layout constraints, and specific UI elements
st.markdown("""
    <style>
    /* Force Pure Black Background */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Custom maximum width for a sleek, centered dashboard */
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 10rem;
    }
    
    /* Hide Streamlit Header and Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style the upload box */
    .stFileUploader {
        background-color: #111111;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333333;
    }

    /* Customize the Run button */
    div.stButton > button {
        background-color: #0d131a !important; 
        color: #cce0ff !important; 
        border: 1px solid #2d4a6b !important; 
        transition: all 0.2s ease-in-out;
    }
    
    /* Button Hover Effect */
    div.stButton > button:hover {
        background-color: #16212e !important; 
        border: 1px solid #5282bd !important; 
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- 2. Main Header & Informative Sections ---

# Header & Intro
st.markdown("""
<div style="text-align: center; margin-top: 20px; margin-bottom: 40px;">
    <h1 style="font-size: 45px; margin-bottom: 10px;">Respiratory Sound Analysis</h1>
    <p style="font-size: 18px; color: #cccccc; max-width: 800px; margin: 0 auto;">
        A deep learning diagnostic tool that classifies respiratory audio into 8 distinct categories:<br>
        <span style="color: #ffffff; font-weight: bold;">Asthma, Bronchiectasis, Bronchiolitis, COPD, Healthy, LRTI, Pneumonia, and URTI.</span>
    </p>
</div>
""", unsafe_allow_html=True)

# How the AI Works
st.markdown((
    '<div style="font-size: 22px; font-weight: bold; margin-bottom: 15px; color: #ffffff; position: relative; z-index: 10;">How the AI Works</div>'
    '<div style="background-color: #111111; padding: 25px; border-radius: 10px; border: 1px solid #333333; margin-bottom: 20px; position: relative;">'
        '<p style="color: #cccccc; margin-top: 0; margin-bottom: 20px; font-size: 15px;">The system standardizes every uploaded audio clip and extracts three distinct acoustic features:</p>'
        
        '<div style="margin-bottom: 12px; display: flex; align-items: flex-start; background-color: #000000; padding: 15px; border-radius: 8px; border: 1px solid #222222;">'
            '<div style="margin-right: 15px; margin-top: 2px;">'
                '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>'
            '</div>'
            '<div>'
                '<div style="margin: 0 0 5px 0; color: #ffffff; font-size: 16px; font-weight: bold;">MFCC</div>'
                '<p style="margin: 0; color: #999999; font-size: 14px; line-height: 1.4;">Captures the physical shape of the vocal tract and airways. This is critical because respiratory diseases physically alter these pathways, changing the unique "timbre" of a breath. MFCCs isolate these subtle textural shifts to instantly distinguish between healthy and obstructed airflow.</p>'
            '</div>'
        '</div>'
        
        '<div style="margin-bottom: 12px; display: flex; align-items: flex-start; background-color: #000000; padding: 15px; border-radius: 8px; border: 1px solid #222222;">'
            '<div style="margin-right: 15px; margin-top: 2px;">'
                '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 20V10"></path><path d="M12 20V4"></path><path d="M6 20v-6"></path></svg>'
            '</div>'
            '<div>'
                '<div style="margin: 0 0 5px 0; color: #ffffff; font-size: 16px; font-weight: bold;">Chroma STFT</div>'
                '<p style="margin: 0; color: #999999; font-size: 14px; line-height: 1.4;">Analyzes the energy distribution across pitch classes over time. This enables the system to track rhythmic breathing cycles and isolate specific pitch-based anomalies, such as the high-frequency whistling of a wheeze from standard background noise.</p>'
            '</div>'
        '</div>'
        
        '<div style="display: flex; align-items: flex-start; background-color: #000000; padding: 15px; border-radius: 8px; border: 1px solid #222222;">'
            '<div style="margin-right: 15px; margin-top: 2px;">'
                '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><path d="M3 9h18"></path><path d="M9 21V9"></path></svg>'
            '</div>'
            '<div>'
                '<div style="margin: 0 0 5px 0; color: #ffffff; font-size: 16px; font-weight: bold;">Mel-Spectrogram</div>'
                '<p style="margin: 0; color: #999999; font-size: 14px; line-height: 1.4;">Maps audio frequencies visually over time using the Mel scale. Since Convolutional Neural Networks (CNNs) excel at image recognition, converting audio into a 2D heatmap allows the model to observe disease signatures, like the sharp vertical spikes of lung crackles.</p>'
            '</div>'
        '</div>'
    '</div>'
    
    '<div style="background-color: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 5px; margin-bottom: 40px;">'
        '<p style="margin: 0; color: #dddddd; font-size: 16px; line-height: 1.5;">Instead of a single model, the engine uses an ensemble of <strong>3 Convolutional Neural Networks (CNNs)</strong>. Each network analyzes one feature independently, combining their results to achieve maximum accuracy.</p>'
    '</div>'
), unsafe_allow_html=True)

# How to Use Section
st.markdown((
    '<div style="font-size: 22px; font-weight: bold; margin-bottom: 15px; color: #ffffff; position: relative; z-index: 10;">How to Use the Dashboard</div>'
    '<div style="display: flex; gap: 15px; margin-bottom: 25px;">'
        '<div style="flex: 1; background-color: #111111; padding: 20px; border-radius: 10px; border: 1px solid #333333;">'
            '<div style="margin-top: 0px; margin-bottom: 10px; color: #ffffff; font-size: 16px; font-weight: bold;">1. Upload</div>'
            '<p style="font-size: 14px; color: #cccccc; margin: 0; line-height: 1.4;">Upload a patient\'s breath recording (<b>.wav</b> file) into the data input box below.</p>'
        '</div>'
        '<div style="flex: 1; background-color: #111111; padding: 20px; border-radius: 10px; border: 1px solid #333333;">'
            '<div style="margin-top: 0px; margin-bottom: 10px; color: #ffffff; font-size: 16px; font-weight: bold;">2. Analyze</div>'
            '<p style="font-size: 14px; color: #cccccc; margin: 0; line-height: 1.4;">Click the analysis button to initialize the AI engine and process the audio features.</p>'
        '</div>'
        '<div style="flex: 1; background-color: #111111; padding: 20px; border-radius: 10px; border: 1px solid #333333;">'
            '<div style="margin-top: 0px; margin-bottom: 10px; color: #ffffff; font-size: 16px; font-weight: bold;">3. Review</div>'
            '<p style="font-size: 14px; color: #cccccc; margin: 0; line-height: 1.4;">View the Diagnosis and the transparent breakdown of what each model predicted.</p>'
        '</div>'
    '</div>'
), unsafe_allow_html=True)

st.markdown("---") 

# --- 3. Backend Logic & Model Loading ---

DISEASE_CLASSES = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'LRTI', 'Pneumonia', 'URTI']

@st.cache_resource
def load_my_models():
    """Loads the pre-trained Keras models with a patch for Dense layer config."""
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    # Patch for potential version mismatch issues during .h5 model loading
    original_dense_from_config = tf.keras.layers.Dense.from_config
    
    @classmethod
    def patched_dense_from_config(cls, config):
        config.pop('quantization_config', None)
        return original_dense_from_config.__func__(cls, config)
        
    tf.keras.layers.Dense.from_config = patched_dense_from_config

    try:
        combined = load_model('model/model.h5', compile=False)
        mfcc = load_model('mfccTrained1.h5', compile=False)
        chroma = load_model('chromaTrained1.h5', compile=False)
        mspec = load_model('mSpecTrained1.h5', compile=False)
    finally:
        tf.keras.layers.Dense.from_config = original_dense_from_config
        
    return combined, mfcc, chroma, mspec

# Initialize models on app startup
try:
    combined_model, mfcc_model, chroma_model, mspec_model = load_my_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# --- 4. Audio Processing Functions ---

def extract_features(audio_data, sample_rate):
    """Pads/truncates audio to 6 seconds and extracts feature arrays."""
    reqLen = 6 * sample_rate 
    
    if len(audio_data) > reqLen:
        audio_data = audio_data[:reqLen]
    else:
        audio_data = librosa.util.pad_center(data=audio_data, size=reqLen)
        
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    mspec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    
    return mfcc, chroma, mspec

def get_prediction_label(predictions):
    """Returns the highest probability disease label and score."""
    pred_probs = predictions.flatten()
    max_index = np.argmax(pred_probs)
    return DISEASE_CLASSES[max_index], pred_probs[max_index]

def get_top_two_predictions(predictions):
    """Returns the top 2 highest probability disease labels and their scores."""
    pred_probs = predictions.flatten()
    top_two_indices = np.argsort(pred_probs)[-2:][::-1]
    
    top1_label = DISEASE_CLASSES[top_two_indices[0]]
    top1_conf = pred_probs[top_two_indices[0]]
    
    top2_label = DISEASE_CLASSES[top_two_indices[1]]
    top2_conf = pred_probs[top_two_indices[1]]
    
    return (top1_label, top1_conf), (top2_label, top2_conf)


# --- 5. Single Page Layout for Upload & Results ---

st.subheader("Data Input")
uploaded_file = st.file_uploader("Upload a patient's .wav audio file", type=['wav'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Run Diagnostic Analysis", use_container_width=True, type="primary"):
        if not models_loaded:
            st.error("Models failed to load. Please check your terminal for errors.")
        else:
            with st.spinner('Processing audio features and running models...'):
                try:
                    # Read and process audio
                    audio_bytes = uploaded_file.getvalue()
                    audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=22050)
                    
                    # Extract features
                    mfcc_feat, chroma_feat, mspec_feat = extract_features(audio_data, sample_rate)
                    
                    # Run the combined model
                    mfcc_test = np.array([mfcc_feat])
                    cstft_test = np.array([chroma_feat])
                    mspec_test = np.array([mspec_feat])
                    
                    final_prediction = combined_model.predict({
                        "mfcc": mfcc_test,
                        "croma": cstft_test, 
                        "mspec": mspec_test
                    })
                    
                    top1, top2 = get_top_two_predictions(final_prediction)
                    
                    # Run individual models for UI breakdown
                    mfcc_ind = np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=-1)
                    chroma_ind = np.expand_dims(np.expand_dims(chroma_feat, axis=0), axis=-1)
                    mspec_ind = np.expand_dims(np.expand_dims(mspec_feat, axis=0), axis=-1)
                    
                    mfcc_pred = mfcc_model.predict(mfcc_ind)
                    chroma_pred = chroma_model.predict(chroma_ind)
                    mspec_pred = mspec_model.predict(mspec_ind)
                    
                    mfcc_top1, mfcc_top2 = get_top_two_predictions(mfcc_pred)
                    chroma_top1, chroma_top2 = get_top_two_predictions(chroma_pred)
                    mspec_top1, mspec_top2 = get_top_two_predictions(mspec_pred)

                    # --- Render Dashboard ---
                    
                    # Custom Sleek Mint Green Success Banner
                    st.markdown("""
                    <div style='background-color: rgba(167, 243, 208, 0.05); border: 1px solid rgba(167, 243, 208, 0.15); padding: 15px; border-radius: 8px; margin-bottom: 30px; display: flex; justify-content: center; align-items: center;'>
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#a7f3d0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style='margin-right: 10px;'>
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        <span style='color: #a7f3d0; font-size: 18px; font-weight: bold; letter-spacing: 0.5px;'>Diagnostic Analysis Complete</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("Diagnostic Results")
                    
                    # Display primary results side-by-side
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.metric(label="Primary Diagnosis", value=top1[0])
                        st.markdown(f"<div style='color: #f43f5e; font-size: 16px; font-weight: bold; margin-top: -15px;'>{top1[1]*100:.2f}% Confidence</div>", unsafe_allow_html=True)
                        
                    with res_col2:
                        st.metric(label="Secondary Diagnosis", value=top2[0])
                        st.markdown(f"<div style='color: #fdba74; font-size: 16px; font-weight: bold; margin-top: -15px;'>{top2[1]*100:.2f}% Confidence</div>", unsafe_allow_html=True)
                    
                    # Display individual model breakdown
                    st.markdown("#### Individual Model Breakdown")
                    m1, m2, m3 = st.columns(3) 
                    
                    with m1:
                        st.markdown(f"""
<div style='background-color: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid #222;'>
    <div style='color: #888888; font-size: 12px; font-weight: bold; letter-spacing: 1px; margin-bottom: 12px;'>MFCC MODEL</div>
    <div style='display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 5px;'>
        <span style='color: #888888; font-size: 13px;'>1st Guess</span>
        <span style='color: #dddddd; font-size: 15px; font-weight: bold;'>{mfcc_top1[0]} <span style='color: #38bdf8; font-size: 13px;'>{mfcc_top1[1]*100:.1f}%</span></span>
    </div>
    <div style='background-color: #1a1a1a; border-radius: 10px; height: 6px; width: 100%; margin-bottom: 12px;'>
        <div style='background: linear-gradient(90deg, #1e3a8a, #38bdf8); width: {mfcc_top1[1]*100:.1f}%; height: 100%; border-radius: 10px;'></div>
    </div>
    <div style='display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 5px;'>
        <span style='color: #666666; font-size: 12px;'>2nd Guess</span>
        <span style='color: #aaaaaa; font-size: 13px; font-weight: bold;'>{mfcc_top2[0]} <span style='color: #888888; font-size: 12px;'>{mfcc_top2[1]*100:.1f}%</span></span>
    </div>
    <div style='background-color: #1a1a1a; border-radius: 10px; height: 4px; width: 100%;'>
        <div style='background: #555555; width: {mfcc_top2[1]*100:.1f}%; height: 100%; border-radius: 10px;'></div>
    </div>
</div>
""", unsafe_allow_html=True)
                        
                    with m2:
                        st.markdown(f"""
<div style='background-color: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid #222;'>
    <div style='color: #888888; font-size: 12px; font-weight: bold; letter-spacing: 1px; margin-bottom: 12px;'>CHROMA MODEL</div>
    <div style='display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 5px;'>
        <span style='color: #888888; font-size: 13px;'>1st Guess</span>
        <span style='color: #dddddd; font-size: 15px; font-weight: bold;'>{chroma_top1[0]} <span style='color: #a855f7; font-size: 13px;'>{chroma_top1[1]*100:.1f}%</span></span>
    </div>
    <div style='background-color: #1a1a1a; border-radius: 10px; height: 6px; width: 100%; margin-bottom: 12px;'>
        <div style='background: linear-gradient(90deg, #5b21b6, #a855f7); width: {chroma_top1[1]*100:.1f}%; height: 100%; border-radius: 10px;'></div>
    </div>
    <div style='display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 5px;'>
        <span style='color: #666666; font-size: 12px;'>2nd Guess</span>
        <span style='color: #aaaaaa; font-size: 13px; font-weight: bold;'>{chroma_top2[0]} <span style='color: #888888; font-size: 12px;'>{chroma_top2[1]*100:.1f}%</span></span>
    </div>
    <div style='background-color: #1a1a1a; border-radius: 10px; height: 4px; width: 100%;'>
        <div style='background: #555555; width: {chroma_top2[1]*100:.1f}%; height: 100%; border-radius: 10px;'></div>
    </div>
</div>
""", unsafe_allow_html=True)
                        
                    with m3:
                        st.markdown(f"""
<div style='background-color: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid #222;'>
    <div style='color: #888888; font-size: 12px; font-weight: bold; letter-spacing: 1px; margin-bottom: 12px;'>MEL SPEC MODEL</div>
    <div style='display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 5px;'>
        <span style='color: #888888; font-size: 13px;'>1st Guess</span>
        <span style='color: #dddddd; font-size: 15px; font-weight: bold;'>{mspec_top1[0]} <span style='color: #10b981; font-size: 13px;'>{mspec_top1[1]*100:.1f}%</span></span>
    </div>
    <div style='background-color: #1a1a1a; border-radius: 10px; height: 6px; width: 100%; margin-bottom: 12px;'>
        <div style='background: linear-gradient(90deg, #064e3b, #10b981); width: {mspec_top1[1]*100:.1f}%; height: 100%; border-radius: 10px;'></div>
    </div>
    <div style='display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 5px;'>
        <span style='color: #666666; font-size: 12px;'>2nd Guess</span>
        <span style='color: #aaaaaa; font-size: 13px; font-weight: bold;'>{mspec_top2[0]} <span style='color: #888888; font-size: 12px;'>{mspec_top2[1]*100:.1f}%</span></span>
    </div>
    <div style='background-color: #1a1a1a; border-radius: 10px; height: 4px; width: 100%;'>
        <div style='background: #555555; width: {mspec_top2[1]*100:.1f}%; height: 100%; border-radius: 10px;'></div>
    </div>
</div>
""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error during processing: {e}")