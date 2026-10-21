"""Download and hash-check the recorded original COCO assets, never fitted weights."""
import argparse
from pathlib import Path
from urllib.request import urlopen
from runtime import digest, read_json


def fetch(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for item in read_json(Path(__file__).with_name('weights_registry.json')).values():
        path = output / item['file']
        if not path.exists():
            temp = path.with_suffix('.download')
            with urlopen(item['url'], timeout=120) as source, temp.open('xb') as dest:
                for block in iter(lambda: source.read(1024*1024), b''):
                    dest.write(block)
            if digest(temp) != item['sha256']:
                raise ValueError(f'Official asset checksum changed: {item["file"]}; investigate before use')
            temp.rename(path)
        if digest(path) != item['sha256']:
            raise ValueError(f'Existing file is not the recorded official asset: {path}')
        print(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    fetch(p.parse_args().output)
