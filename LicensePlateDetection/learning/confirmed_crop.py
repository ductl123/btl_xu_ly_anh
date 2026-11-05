"""Read a manually confirmed label for an identical training crop."""
from contextlib import closing
import json
import sqlite3

from .dataset import BASE, image_digest


def confirmed_crop_rows(image, plate_type, db=None):
    database = db if db is not None else BASE / 'data/ocr_training/labels.sqlite3'
    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro',
                                     uri=True, timeout=.25)) as connection:
            rows = connection.execute(
                'SELECT row1,row2,provenance,split FROM examples '
                'WHERE active=1 AND image_sha=? AND plate_type=?',
                (image_digest(image), plate_type),
            ).fetchall()
    except sqlite3.OperationalError:
        return None
    labels = {(first, second) for first, second, _, _ in rows}
    if len(labels) != 1:
        return None
    for _, _, provenance, split in rows:
        if split != 'train':
            continue
        try:
            metadata = json.loads(provenance)
        except (TypeError, ValueError):
            continue
        if isinstance(metadata, dict) and (
                metadata.get('type') == 'user_confirmed' or metadata.get('user_correction')):
            return labels.pop()
    return None
