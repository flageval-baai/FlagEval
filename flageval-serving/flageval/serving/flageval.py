import io
import os
from dataclasses import dataclass, field
from typing import List, Dict

import requests
import tqdm
import sys


@dataclass
class Chunk:
    num: int
    size_kb: int
    ks3_ready: bool = False


@dataclass
class File:
    path: str
    filename: str
    size_kb: int
    id_: int = 0
    ks3_ready: bool = False
    chunks: List[Chunk] = field(default_factory=list)


class FlagEvalError(Exception):
    pass

def is_empty(string):
    if string is None or string == "":
        return True
    else:
        return False
    

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
        for root, dirs, files in os.walk(
                self.path, followlinks=self.options.get('followlinks', False),
        ):
            for name in files:
                path = os.path.join(root, name)
                filename = path[len(self.path):]
                st_size = os.stat(path).st_size
                results.append(File(path, filename, max(int(st_size / 1024), 1)))
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

    def _do_upload(self, item: File, progbar: tqdm.tqdm, is_update = 0):
        url = f'{self.host}{self.CHUNKS_PATH.format(item.id_)}'
        innerbar = tqdm.tqdm(total=item.size_kb, unit='KB', desc=item.filename)

        with open(item.path,'rb') as f:
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
                    'is_update': is_update
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

class FlagEvalManager(FlagEvalUploader):
    FILES_PATH1 = "/api/evaluations/manage/op_file"

    def __init__(self, host: str, token: str, src_path: str, dst_path: str, **kwargs):
        self.host = host.rstrip('/')
        self.src_path = src_path
        self.dst_path = dst_path
        self.token = token
        self.options = kwargs

    def list_remote_files(self, path, opt = 1):
        path = self.filter_path(path)
        results: List[File] = []
        url = f'{self.host}{self.FILES_PATH1}'
        resp = requests.post(url, data={"command": "ls", 'token': self.token, "src_path": path, "opt": opt})
        resp = resp.json()
        for item in resp['results']:
            path = item['filename']
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
        return results, resp["canSendTargets"]
    
    def filter_path(self, path):
        if path.find(".") == 0:
            path = path[1:]
        cur = 0
        while cur < len(path) and path[cur] == '/':
            cur += 1
        path = path[cur:]
        while len(path) > 0 and path[-1] == "/":
            path = path[:-1]
        return path
    
    def rm(self):
        if is_empty(self.src_path):
            print("error: file path is empty", file=sys.stderr)
            return
        self.src_path = self.filter_path(self.src_path)
        url = f'{self.host}{self.FILES_PATH1}'
        resp = requests.post(url, data = {"command": "rm", "token": self.token, "src_path": self.src_path})
        if resp.status_code == 200:
            data = resp.json()
            ret = data["ret"]
            if ret != 0:
                print("service return error:", ret)
        else:
            print("service error:", resp.status_code, file=sys.stderr)
    
    def cp(self):
        if is_empty(self.src_path):
            print("error: file path is empty", file=sys.stderr)
            return
        if is_empty(self.dst_path):
            self.dst_path = "/"
        self.src_path = os.path.normpath(self.src_path)
        self.dst_path = self.filter_path(self.dst_path)
        local_files = self._list_local_files()
        if len(local_files) == 0:
            return

        a = self._create_remote_files(local_files)
        remote_files = []
        for i in range(len(a["results"])):
            chunks = []
            item = a["results"][i]
            for item1 in item["chunks"]:
                chunks.append(Chunk(item1["num"], item1["sizeKb"], item1["ks3Ready"]))
            remote_files.append(File(local_files[i].path, item["filename"], item["sizeKb"], item["id"], item["ks3Ready"], chunks))

        total_kb = sum(x.size_kb for x in remote_files)
        progbar = tqdm.tqdm(total=total_kb, unit="KB")

        for item in remote_files:
            if item.ks3_ready:
                progbar.update(item.size_kb)
                continue
            self._do_upload(item, progbar, 1)
        progbar.close()
    
    def _list_local_files(self) -> List[File]:
        results: List[File] = []
        rfiles, canSendTargets = self.list_remote_files(self.dst_path, 2)
        if os.path.isfile(self.src_path):
            if len(rfiles) == 0 or len(rfiles) == 1:
                if len(self.dst_path) == 0:
                    filename = self.src_path
                else:
                    filename = self.dst_path
            else:
                filename = os.path.join(self.dst_path, os.path.basename(self.src_path))
            st_size = os.stat(self.src_path).st_size
            if canSendTargets or filename.find("targets/") != 0:
                results.append(File(self.src_path, filename, max(int(st_size / 1024), 1)))
            else:
                print("warning: targets/ can not be changed, ignoring file("+self.src_path+")")
            return results
        elif os.path.isdir(self.src_path):
            abs_path = os.path.abspath(self.src_path)
            #if len(rfiles) > 0 or self.dst_path == "":
            if len(rfiles) > 0:
                basename = os.path.basename(self.src_path)
            else:
                basename = ""
            for root, dirs, files in os.walk(
                    abs_path, followlinks=self.options.get('followlinks', False),
            ):
                for name in files:
                    path = os.path.join(root, name)
                    filename = os.path.join(self.dst_path, basename, path[len(abs_path) + 1:])
                    st_size = os.stat(path).st_size
                    if canSendTargets or filename.find("targets/") != 0:
                        results.append(File(path, filename, max(int(st_size / 1024), 1)))
                    else:
                        print("warning: targets/ can not be changed, ignoring file("+path+")")
            if len(results) == 0:
                print("error: skip upload because directory {"+self.src_path+"} is empty.", file=sys.stderr)
        else:
            print("error: " + self.src_path + " is not exist", file=sys.stderr)
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
