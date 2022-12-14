# coding: utf-8

import os
import json
import uuid
import shutil
import zipfile

TEMP_PATH = '/tmp'


def mkdirp(path):
    """ mkdirp
    """
    if not os.path.exists(path):
        os.mkdir(path)


def write_file(path, content):
    """ write_file
    """
    with open(path, 'w') as f:
        f.write(content)


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def replace_file_content(path, source, target):
    content = read_file(path)
    write_file(path, content.replace(source, target))


def unshift_file_content(path, append_content):
    content = read_file(path)
    write_file(path, append_content + content)


def append_file_content(path, append_content):
    content = read_file(path)
    write_file(path, content + append_content)


def read_json_file(path):
    with open(path, 'r') as f:
        return json.load(f)


def create_temp_dir():
    dir_name = os.path.join(TEMP_PATH, str(uuid.uuid4()))
    os.mkdir(dir_name)
    return dir_name


def remove_temp_dir(dir_path):
    if dir_path.startswith(TEMP_PATH) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    else:
        raise Exception(dir_path)


def unzip(path, target):
    fp = zipfile.ZipFile(path, 'r')
    fp.extractall(target)
    fp.close()


def text_to_temp_file(text, ext='.txt'):
    file_name = os.path.join(TEMP_PATH, str(uuid.uuid4()) + ext)
    with open(file_name, 'w') as f:
        f.write(text)
    return file_name


def get_dir_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += (os.path.getsize(fp) if os.path.isfile(fp) else 0)
    return total_size
