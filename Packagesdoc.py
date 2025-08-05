# Run this cell first to install required packages
!pip install wfdb
!pip install requests
!pip install tqdm

# Verify installation
try:
    import wfdb
    import requests
    import tqdm
    print("✅ All packages installed successfully!")
    print("You can now run the main MIT-BIH dataset acquisition script.")
except ImportError as e:
    print(f"❌ Installation failed: {e}")
    print("Please restart runtime and try again.")
