"""Explicit setup download; inference never downloads or uploads images."""
from pathlib import Path
import hashlib,io,json,os,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]/'models/line_ocr'

def main():
    metadata=json.loads((ROOT/'pretrained_metadata.json').read_text(encoding='utf-8'))
    target=ROOT/'english_g2.pth'
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==metadata['sha256']:
        print('Pretrained weights already verified');return
    url=metadata['model_url']
    if not url.startswith('https://github.com/JaidedAI/EasyOCR/releases/'):
        raise ValueError('Unexpected model source URL')
    request=urllib.request.Request(url,headers={'User-Agent':'LicensePlateDetection-setup'})
    with urllib.request.urlopen(request,timeout=60) as response:archive=response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:payload=zipped.read('english_g2.pth')
    if hashlib.sha256(payload).hexdigest()!=metadata['sha256']:raise ValueError('Downloaded weights failed integrity verification')
    temporary=target.with_suffix('.download');temporary.write_bytes(payload);os.replace(temporary,target)
    print('Downloaded and verified english_g2.pth')

if __name__=='__main__':main()
