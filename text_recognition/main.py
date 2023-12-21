from pathlib import Path

from ocr_model import OCRModel
from evaluation import evaluate_model
import os, easyocr

DATA_PATH = "dataset/"

def main():

    model = OCRModel()
    data_path = Path(__file__).parent.parent / 'data' / 'public_data'
    accuracy = evaluate_model(model, data_path)
    print(f'Final accuracy is {accuracy:.3f}')


if __name__ == '__main__':
    main()
