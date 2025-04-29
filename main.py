from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List
import easyocr
import cv2
import numpy as np
import re
import os
import uvicorn
class LabTest(BaseModel):
    test_name: str
    test_value: str
    bio_reference_range: str
    test_unit: str
    lab_test_out_of_range: bool

app = FastAPI(title="Lab Test Extraction API")

reader = easyocr.Reader(['en'])

def preprocess_image(image_bytes):

    nparr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opening

def extract_lab_tests(text_results):
    lab_tests = []

    patterns = [
        r"([A-Za-z\s\(\)]+):\s*([0-9\.]+)\s*([a-zA-Z/%]+)?\s*\(([0-9\.]+)\s*-\s*([0-9\.]+)\)",

        r"([A-Za-z\s\(\)]+)\s+([0-9\.]+)\s+([a-zA-Z/%]+)\s+([0-9\.]+)\s*-\s*([0-9\.]+)"
    ]
    
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
        
        <div id="rawTextSection">
            <h2 class="section-title">Recognized Raw Text:</h2>
            <pre id="rawText">Upload an image to see the raw text...</pre>
        </div>
        
        <div id="results">
            <h2 class="section-title">Extracted Lab Test Data:</h2>
            <pre id="jsonResult">Upload an image to see processed results...</pre>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('file');
                formData.append('file', fileInput.files[0]);
                
                try {
                    document.getElementById('rawText').textContent = "Processing...";
                    document.getElementById('jsonResult').textContent = "Processing...";
                    
                    const response = await fetch('/get-lab-tests', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    // Display the raw recognized text
                    document.getElementById('rawText').textContent = data.recognized_text.join('\\n');
                    
                    // Display the processed results
                    document.getElementById('jsonResult').textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('rawText').textContent = "Error processing image";
                    document.getElementById('jsonResult').textContent = "Error processing image";
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

    try:        
        contents = await file.read()

        processed_img = preprocess_image(contents)
        results = reader.readtext(processed_img, detail=0)
        lab_tests = extract_lab_tests(results)
        return {
            "is_success": True,
            "recognized_text": results,  
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

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )
