import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *
import zipfile

def main():
    archive=ROOT/'analysis'/'supervisor_analyses_2026-10-13.zip'
    assert read('output_validation.json')['passed']
    assert read('original_artifact_preservation.json')['all_unchanged']
    files=[]
    for directory in [ROOT/'audit',ROOT/'src',OUT]:
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and 'runtime' not in p.parts:
                files.append(p)
    files += [p for p in (ROOT/'results').glob('*.csv') if not p.name.startswith('ieee_cis_graph_edges_')]
    files += [ROOT/'README.md',ROOT/'notes.txt',ROOT/'notes2.txt',ROOT/'requirements.txt']
    files=sorted(set(files)-{OUT/'package_contents.json'},key=str)
    package_contents=[{'archive_path':str(p.relative_to(ROOT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    dump('package_contents.json',package_contents)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[OUT/'package_contents.json']:
            z.write(p,str(p.relative_to(ROOT)).replace('\\','/'))
        z.writestr('START_HERE.md','# Supervisor analysis package\n\nOpen [the analysis README](analysis/supervisor_2026-10-13/README.md).\n\nIncludes the complete earlier audit tree and all new analyses. Original datasets, tensor graph caches and checkpoints are external; paths and hashes are recorded. See the analysis README for reproducibility requirements.\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in package_contents:
            assert hashlib.sha256(z.read(item['archive_path'])).hexdigest()==item['sha256']
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    print('Verified ZIP',archive,'files',len(files)+2,'bytes',archive.stat().st_size,flush=True)

if __name__=='__main__':main()
