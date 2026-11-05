# Labelled character reference bank

Each class directory contains binary PNG masks: white is ink, black is background.
Labels are explicit, not predictions. `manifest.json` records the source crop,
bounding box and label provenance. The current bank contains 32 manually verified glyphs across 17 classes.
It includes the initial 19 glyphs from 18-MD2 074.73 and 37-AC 171.24,
plus references from examples documented in loi_sai.docx. See manifest.json
for every label and source.

These source photos are calibration data, not an independent accuracy test. No plate
strings or image filenames are used by the recognizer at inference. A reference
can match the same character in any photo.

To expand coverage, verify the correct character by eye, save its tight binary
crop to `<class>/<unique name>.png`, record provenance in the manifest, then
restart the GUI. Do not auto-label new references from OCR output. Keep all crops
and augmentations from one source photo in the same data split. Do not put test
photo glyphs into this directory when reporting held-out accuracy.

30 classes: 0123456789ABCDEFGHKLMNPSTUVXYZ.

The two additional 8 masks come from 29-G1 727.58, explicitly confirmed by the user on 2026-11-05. Both variants belong to the same calibration source, not two independent test photos.
