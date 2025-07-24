# download_script.py
import nltk
import os
import time

# Define the target directory for NLTK data within your repository
# This will put the data inside the 'nltk_data' folder you created
download_target_dir = os.path.join(os.getcwd(), 'nltk_data')

# Ensure the directory exists (it should, we created it earlier)
if not os.path.exists(download_target_dir):
    os.makedirs(download_target_dir)

    # Tell NLTK to download to this specific directory for this script's session
if download_target_dir not in nltk.data.path:
   nltk.data.path.insert(0, download_target_dir)

print(f"Attempting to download NLTK data to: {download_target_dir}")

try:
            # Download 'stopwords'
    print("Downloading 'stopwords'...")
    nltk.download('stopwords', download_dir=download_target_dir)
    print("'stopwords' downloaded successfully.")

                            # Download 'wordnet'
    print("Downloading 'wordnet'...")
    nltk.download('wordnet', download_dir=download_target_dir)
    print("'wordnet' downloaded successfully.")

    print("\nNLTK data download complete! Verifying structure...")

                                                # --- VERIFICATION ---
    stopwords_path = os.path.join(download_target_dir, 'corpora', 'stopwords')
    wordnet_path = os.path.join(download_target_dir, 'corpora', 'wordnet')

    if os.path.exists(stopwords_path) and os.listdir(stopwords_path):
        print(f"✅ 'stopwords' folder found at: {stopwords_path}")
    else:
        print(f"❌ ERROR: 'stopwords' folder NOT found or empty at: {stopwords_path}")

    if os.path.exists(wordnet_path) and os.listdir(wordnet_path):
        print(f"✅ 'wordnet' folder found at: {wordnet_path}")
    else:
        print(f"❌ ERROR: 'wordnet' folder NOT found or empty at: {wordnet_path}")

    print("\nIf both show '✅', you can proceed to Phase 2: Add NLTK Data to GitHub.")

except Exception as e: 
           print(f"\nCRITICAL ERROR during NLTK download in Codespaces: {e}")
           print("Please ensure your Codespace has internet access.")
           