import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import requests
import tqdm


@dataclass
class Chunk:
    num: int
    size_kb: int
    ks3_ready: bool = False


@dataclass
class File:
    path: Path
    filename: str
    size_kb: int
    id_: int = 0
    ks3_ready: bool = False
    chunks: List[Chunk] = field(default_factory=list)


class FlagEvalError(Exception):
    pass


class FlagEvalUploader:
    FILES_PATH = "/api/evaluations/models/files"
    CHUNKS_PATH = "/api/evaluations/models/files/{}/chunks"

    def __init__(self, host: str, token: str, path: str, **kwargs):
        self.host = host.rstrip('/')
        self.token = token
        if not path.endswith("/"):
            path = f'{path}/'
        self.path = Path(path).absolute()
        self.options = kwargs

    def upload(self):
        local_files = self._list_local_files()
        remote_files = self._list_remote_files()
        if len(remote_files) == 0:
            self._create_remote_files(local_files)
            remote_files = self._list_remote_files()

        total_kb = sum(x.size_kb for x in remote_files)
        progbar = tqdm.tqdm(total=total_kb, unit="KB")

        for item in remote_files:
            if item.ks3_ready:
                progbar.update(item.size_kb)
                continue
            self._do_upload(item, progbar)
        progbar.close()

    def _list_local_files(self) -> List[File]:
        results: List[File] = []
        for root, _dirs, files in os.walk(
                self.path, followlinks=self.options.get('followlinks', False),
        ):
            for name in files:
                path = Path(os.path.join(root, name)).absolute()
                # +1 for the seperator /(Posix) or \(Windows)
                filename = path.as_posix()[len(self.path.as_posix()) + 1:]
                st_size = os.stat(path).st_size
                results.append(File(path, filename, max(int(st_size / 1024), 1)))
        return results

    def _list_remote_files(self) -> List[File]:
        results: List[File] = []
        url = f'{self.host}{self.FILES_PATH}'
        resp = requests.get(url, params={'token': self.token})

        for item in resp.json()['results']:
            path = Path(os.path.join(self.path, item['filename']))
            results.append(File(
                path=path,
                filename=item['filename'],
                size_kb=item['sizeKb'],
                id_=item["id"],
                ks3_ready=item["ks3Ready"],
                chunks=[
                    Chunk(
                        num=x['num'],
                        size_kb=x['sizeKb'],
                        ks3_ready=x['ks3Ready'],
                    )
                    for x in item['chunks']
                ]
            ))
        return results

    def _create_remote_files(self, local_files: List[File]):
        url = f'{self.host}{self.FILES_PATH}'
        resp = requests.post(url, json={
            'token': self.token,
            'files': [
                {
                    'filename': item.filename,
                    'sizeKb': item.size_kb,
                }
                for item in local_files
            ]
        })
        if resp.status_code >= 400:
            raise FlagEvalError(resp.status_code, resp.text)
        return resp.json()

    def _do_upload(self, item: File, progbar: tqdm.tqdm):
        url = f'{self.host}{self.CHUNKS_PATH.format(item.id_)}'
        innerbar = tqdm.tqdm(total=item.size_kb, unit='KB', desc=item.filename)

        with open(item.path, 'rb') as f:
            for i, chunk in enumerate(item.chunks):
                buf = io.BytesIO()
                if i + 1 < len(item.chunks):
                    content = f.read(chunk.size_kb * 1024)
                else:
                    content = f.read()
                buf.write(content)
                buf.seek(0)
                if chunk.ks3_ready:
                    progbar.update(chunk.size_kb)
                    innerbar.update(chunk.size_kb)
                    continue

                files = {'file': buf}
                values = {
                    'chunk_num': chunk.num,
                    'token': self.token,
                }
                resp = requests.post(
                    url, files=files, params=values,
                    headers={
                        'Content-Disposition': f'attachment; filename="{item.filename}"',
                    },
                )
                if resp.status_code >= 400:
                    raise FlagEvalError(resp.status_code, resp.text)
                r = resp.json()
                if r['status'] != 200:
                    raise FlagEvalError(r)

                progbar.update(chunk.size_kb)
                innerbar.update(chunk.size_kb)
        innerbar.close()
