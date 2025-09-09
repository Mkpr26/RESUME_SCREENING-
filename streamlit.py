import streamlit as st
import pickle
import re
import string

# ---------------------------
# Load your trained model + TF-IDF vectorizer
# ---------------------------
model = pickle.load(open("resume_model.pkl", "rb"))       # your trained ML model
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))  # fitted TfidfVectorizer

# ---------------------------
# Cleaning function (same as before)
# ---------------------------
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove urls
    resumeText = re.sub('RT|cc', ' ', resumeText)       # remove RT/cc
    resumeText = re.sub('#\S+', '', resumeText)         # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)       # remove mentions
    resumeText = re.sub('[%s]' % re.escape(string.punctuation), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)  # remove non-ascii
    resumeText = re.sub('\s+', ' ', resumeText)         # remove extra spaces
    return resumeText.lower()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📄 Resume Screening App (ML-powered)")
st.write("Upload a resume and let the ML model predict the job category!")

# Option 1: Text input
resume_text = st.text_area("Paste the resume text here:")


# Option 2: File upload
uploaded_file = st.file_uploader("Or upload a resume (.txt file)", type=["txt"])
if uploaded_file is not None:
    resume_text = uploaded_file.read().decode("utf-8")

if st.button("Predict Category"):
    if resume_text:
        # Clean and transform
        cleaned = cleanResume(resume_text)
        vectorized = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        
        # Show result
        st.success(f"✅ Predicted Category: **{prediction}**")
    else:
        st.warning("⚠️ Please enter or upload a resume first.")
