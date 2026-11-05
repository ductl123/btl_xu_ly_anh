# Plate object verifier

The model uses a NumPy spatial HOG descriptor and a regularized linear classifier.
It helps distinguish a plate crop from roofs, walls, tires, shelves and signs.
Its signed margin is evidence, not a probability or recognition accuracy.

All crops in manifest.json are explicitly reviewed calibration data. They were
selected from the user's development photographs and error document. There is
currently no independent held-out evaluation set. Augmentations from a source
are training examples, not additional independent photographs.

Rebuild: run the project's Python with tools/train_plate_verifier.py.
Only reviewed object labels belong here. Never auto-label OCR guesses.
The classifier supplements geometry and character-row checks; it is not a general
vehicle detector and cannot recover characters destroyed by glare or blur.
