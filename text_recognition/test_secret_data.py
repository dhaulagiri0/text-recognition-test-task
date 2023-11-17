from pathlib import Path

from text_recognition.ocr_model import OCRModel
from text_recognition.evaluation import evaluate_model


# Don't use or change this file, I ll use it to run your model on secret dataset easily
def main():
    model = OCRModel()
    data_path = Path.home() / 'secret_data'
    accuracy = evaluate_model(model, data_path)
    print(f'Final accuracy is {accuracy:.3f}')


if __name__ == '__main__':
    main()
