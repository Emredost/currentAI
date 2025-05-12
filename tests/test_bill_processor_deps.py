"""
Test script to verify bill processor dependencies are installed correctly.
"""
import pytest
import importlib
import subprocess
import sys


def test_pdf_processing_dependencies():
    """Test that PDF processing dependencies are available."""
    # Test for pdfplumber
    try:
        importlib.import_module('pdfplumber')
    except ImportError:
        pytest.fail("pdfplumber is not installed. Run 'pip install pdfplumber'")


def test_image_processing_dependencies():
    """Test that image processing dependencies are available."""
    # Test for Pillow
    try:
        importlib.import_module('PIL')
    except ImportError:
        pytest.fail("Pillow is not installed. Run 'pip install Pillow'")
    
    # Test for pytesseract
    try:
        importlib.import_module('pytesseract')
    except ImportError:
        pytest.fail("pytesseract is not installed. Run 'pip install pytesseract'")


def test_tesseract_installation():
    """Test that tesseract OCR engine is installed and available."""
    try:
        import pytesseract
        # Try to get tesseract version
        version = pytesseract.get_tesseract_version()
        
        if version == 0:
            pytest.fail("Tesseract OCR is not available. Please install it.")
            
    except Exception as e:
        pytest.fail(f"Error checking tesseract installation: {str(e)}")


def test_import_modules():
    """Test importing all modules needed for bill processor."""
    required_modules = [
        'io', 
        're', 
        'numpy', 
        'pandas', 
        'datetime',
        'matplotlib',
        'streamlit',
    ]
    
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            pytest.fail(f"Required module {module} is not available")


if __name__ == "__main__":
    # Simple script mode to check dependencies without running pytest
    print("Checking bill processor dependencies...")
    
    # Check for Python packages
    required_packages = ['pdfplumber', 'PIL', 'pytesseract', 'numpy', 
                         'pandas', 'streamlit', 'matplotlib']
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")
    
    # Check for Tesseract OCR
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        if version > 0:
            print(f"✓ Tesseract OCR is installed (version {version})")
        else:
            print("✗ Tesseract OCR is NOT properly installed")
            missing_packages.append("tesseract")
    except Exception as e:
        print(f"✗ Failed to check Tesseract OCR: {str(e)}")
        missing_packages.append("tesseract")
    
    # Print summary
    if missing_packages:
        print("\nMissing dependencies:")
        if 'pdfplumber' in missing_packages:
            print("  - Install pdfplumber: pip install pdfplumber")
        if 'PIL' in missing_packages:
            print("  - Install Pillow: pip install Pillow")
        if 'pytesseract' in missing_packages:
            print("  - Install pytesseract: pip install pytesseract")
        if 'tesseract' in missing_packages:
            print("  - Install Tesseract OCR:")
            print("    - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            print("    - macOS: brew install tesseract")
            print("    - Linux: sudo apt install tesseract-ocr")
        print("\nRun: pip install -r requirements.txt")
    else:
        print("\nAll dependencies are installed correctly!") 