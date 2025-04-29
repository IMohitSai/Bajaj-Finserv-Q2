from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List
import easyocr
import cv2
import numpy as np
import re

# Define data models
class LabTest(BaseModel):
    test_name: str
    test_value: str
    bio_reference_range: str
    test_unit: str
    lab_test_out_of_range: bool

# Initialize FastAPI app
app = FastAPI(title="Lab Test Extraction API")

# Initialize EasyOCR reader (only once, at startup)
reader = easyocr.Reader(['en'])

def preprocess_image(image_bytes):
    """Preprocess the image to improve OCR accuracy"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Noise removal
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opening

def extract_lab_tests(text_results):
    """Extract lab test information from OCR results"""
    lab_tests = []
    
    # Various patterns to match lab test entries
    patterns = [
        # Pattern for "Test Name: Value Unit (Min-Max)"
        r"([A-Za-z\s\(\)]+):\s*([0-9\.]+)\s*([a-zA-Z/%]+)?\s*\(([0-9\.]+)\s*-\s*([0-9\.]+)\)",
        
        # Pattern for "Test Name Value Unit Min-Max"
        r"([A-Za-z\s\(\)]+)\s+([0-9\.]+)\s+([a-zA-Z/%]+)\s+([0-9\.]+)\s*-\s*([0-9\.]+)"
    ]
    
    # Process each line of text
    for line in text_results:
        for pattern in patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                test_name = match[0].strip()
                test_value = match[1].strip()
                test_unit = match[2].strip() if match[2] else ""
                min_range = float(match[3].strip())
                max_range = float(match[4].strip())
                
                bio_reference_range = f"{min_range}-{max_range}"
                
                # Calculate if test is out of range
                try:
                    value_float = float(test_value)
                    lab_test_out_of_range = value_float < min_range or value_float > max_range
                except ValueError:
                    lab_test_out_of_range = False
                
                lab_tests.append(
                    LabTest(
                        test_name=test_name,
                        test_value=test_value,
                        bio_reference_range=bio_reference_range,
                        test_unit=test_unit,
                        lab_test_out_of_range=lab_test_out_of_range
                    )
                )
    
    return lab_tests

@app.get("/", response_class=HTMLResponse)
async def main():
    """Serve the upload form"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lab Report Processor</title>
    </head>
    <body>
        <h1>Lab Report Image Processor</h1>
        <form action="/get-lab-tests" method="post" enctype="multipart/form-data" id="uploadForm">
            <div class="form-group">
                <label for="file">Upload Lab Report Image:</label>
                <input type="file" name="file" id="file" accept="image/*" required>
            </div>
            <button type="submit">Process Image</button>
        </form>
        
        <div id="results">
            <h2>Extracted Lab Test Data:</h2>
            <pre id="jsonResult"></pre>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('file');
                formData.append('file', fileInput.files[0]);
                
                try {
                    const response = await fetch('/get-lab-tests', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    document.getElementById('jsonResult').textContent = JSON.stringify(data, null, 2);
                    document.getElementById('results').style.display = 'block';
                } catch (error) {
                    console.error('Error:', error);
                    alert('An error occurred while processing the image.');
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    """
    Process lab report image and extract test data
    """
    try:
        # Read the file content
        contents = await file.read()
        
        # Preprocess the image
        processed_img = preprocess_image(contents)
        
        # Perform OCR on the image
        results = reader.readtext(processed_img, detail=0)
        
        # Extract lab test data
        lab_tests = extract_lab_tests(results)
        
        # Return the response
        return {
            "is_success": True,
            "data": [test.dict() for test in lab_tests]
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "is_success": False,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
