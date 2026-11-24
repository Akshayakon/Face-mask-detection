# Face Mask Detection

Real-time face mask detection from a webcam. Faces are found with OpenCV's SSD
face detector, then each face is classified as **Mask** or **No Mask** by a
MobileNetV2 model fine-tuned on the images in `Data/`.

Reported accuracy on a held-out 20% test split: about 99.5%. The dataset contains
many near-duplicate images, so real-world accuracy may be lower.

## Project layout

```
MINI_project/
├── src/
│   ├── app.py                    # Live webcam detection app
│   └── train_mask_detector.py    # Trains the classifier
├── models/
│   ├── mask_detector.h5          # Trained mask classifier
│   └── face_detector/            # Pretrained face detector (Caffe)
├── Data/
│   ├── mask/                     # Training images with masks
│   └── no_mask/                  # Training images without masks
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.9 to 3.12. Run all commands from the repository root.

```powershell
python -m venv C:\fmvenv
C:\fmvenv\Scripts\python.exe -m pip install -r requirements.txt
```

On Windows, create the virtual environment in a **short path** such as `C:\fmvenv`.
TensorFlow has very long file names, and a long project path can exceed the
260-character limit and make `pip install` fail.

`opencv-python` must be version 4.x. OpenCV 5 removed support for the Caffe face
detector.

## Run

```powershell
C:\fmvenv\Scripts\python.exe src\app.py
```

A webcam window opens. Click it and press **q** to quit.

## Retrain

```powershell
C:\fmvenv\Scripts\python.exe src\train_mask_detector.py
```

Training reads `Data/mask` and `Data/no_mask`, prints a classification report,
and overwrites `models/mask_detector.h5`. It takes roughly
20 to 40 minutes on a CPU.
