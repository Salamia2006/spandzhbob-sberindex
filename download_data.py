"""Optional re-download. Existing checked-in consumption.parquet works offline."""
import argparse,hashlib,io,json,ssl,urllib.request,zipfile
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--allow-untrusted-sber-certificate',action='store_true',help='Explicit one-request override for Sber certificate chain only')
args=ap.parse_args()
url='https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip'
ctx=ssl._create_unverified_context() if args.allow_untrusted_sber_certificate else ssl.create_default_context()
with urllib.request.urlopen(url,context=ctx,timeout=120) as response:
 raw=response.read()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 content=z.read('hackathonlicence/consumption.parquet')
expected='9833ddaaee7b2a182ed4cceeed16469031700ea87d508bd976150c6770ef8a61'
if hashlib.sha256(content).hexdigest()!=expected:
 raise ValueError('Source changed: inspect the new release before replacing the frozen dataset')
Path('data').mkdir(exist_ok=True);Path('data/consumption.parquet').write_bytes(content)
print('Verified consumption.parquet')
