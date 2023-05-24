import os
from dataclasses import dataclass
from typing import List, Dict

import requests
import tqdm


@dataclass
class File:
    path: str
    filename: str
    size_kb: int
    id_: int = 0
    ks3_ready: bool = False


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
        self.path = path
        self.options = kwargs

    def upload(self):
        local_files = self._list_local_files()
        remote_files = self._list_remote_files()
        if len(remote_files) == 0:
            self._create_remote_files(local_files)
            remote_files = self._list_remote_files()

        for item in tqdm.tqdm(remote_files):
            if not item.ks3_ready:
                r = self._do_upload(item)
                if r['status'] != 200:
                    raise FlagEvalError(r)

    def _list_local_files(self) -> List[File]:
        results: List[File] = []
        for root, dirs, files in os.walk(
                self.path, followlinks=self.options.get('followlinks', False),
        ):
            for name in files:
                path = os.path.join(root, name)
                filename = path[len(self.path):]
                size_kb = int(os.stat(path).st_size / 1024)
                results.append(File(path, filename, size_kb))
        return results

    def _list_remote_files(self) -> List[File]:
        results: List[File] = []
        url = f'{self.host}{self.FILES_PATH}'
        resp = requests.get(url, params={'token': self.token})

        for item in resp.json()['results']:
            path = os.path.join(self.path, item['filename'])
            results.append(File(
                path=path,
                filename=item['filename'],
                size_kb=item['sizeKb'],
                id_=item["id"],
                ks3_ready=item["ks3Ready"],
            ))
        return results

    def _create_remote_files(self, local_files: List[File]):
        url = f'{self.host}{self.FILES_PATH}'
        resp = requests.post(url, json={
            'token': self.token,
            'files': [
                {
                    'filename': item.filename,
                    'size_kb': item.size_kb,
                }
                for item in local_files
            ]
        })
        return resp.json()

    def _do_upload(self, item: File):
        url = f'{self.host}{self.CHUNKS_PATH.format(item.id_)}'
        files = {'file': open(item.path,'rb')}
        values = {
            'chunks_num': '1',
            'part_num': '1',
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
        return resp.json()
