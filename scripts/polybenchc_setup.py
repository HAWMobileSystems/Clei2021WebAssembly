import os
import re
import tarfile
from typing import Pattern, AnyStr

import requests

from polybenchc_common import TMP_DIR, RELATIVE_BENCHMARK_C_SOURCES, BENCHMARK_C_SOURCES

URL = 'https://netix.dl.sourceforge.net/project/polybench/polybench-c-4.2.1-beta.tar.gz'
TAR_GZ_PATH = os.path.join(TMP_DIR, 'polybench-c-4.2.1-beta.tar.gz')
SRC_PATH = os.path.join(TMP_DIR, 'polybench-c-4.2.1-beta')

POLYBENCH_C_FILE = os.path.join(SRC_PATH, 'utilities', 'polybench.c')
POLYBENCH_H_FILE = os.path.join(SRC_PATH, 'utilities', 'polybench.h')
POLYBENCH_TEST_C_FILES = BENCHMARK_C_SOURCES


def download_if_necessary():
    if os.path.exists(TAR_GZ_PATH):
        print('skip download: file already exists')
    else:
        print('downloading...')
        response = requests.get(URL)
        open(TAR_GZ_PATH, 'wb').write(response.content)
    if os.path.exists(SRC_PATH):
        print('skip extraction: directory already exists')
    else:
        print('extracting...')
        tar = tarfile.open(TAR_GZ_PATH, "r:gz")
        tar.extractall(path=TMP_DIR)
        tar.close()


def check_if_patched() -> bool:
    f = open(POLYBENCH_H_FILE, 'r')
    content = f.read()
    f.close()
    return 'testinitonly' in content


def patch(file: str, regex: str, string):
    f = open(file, 'r')
    content = f.read()
    f.close()
    (content, count) = re.subn(regex, string, content)
    if count != 1:
        raise Exception('failed to patch c file')
    f = open(file, 'w')
    f.write(content)
    f.close()


def patch_if_necessary():
    if check_if_patched():
        print('skip patch: files already patched')
    else:
        print('patching...')
        patch(POLYBENCH_H_FILE, r'#endif /\* !POLYBENCH_H \*/',
              r'extern void testinitonly(int argc, char** argv);\n\n\g<0>')
        patch(POLYBENCH_C_FILE, r'\Z', r'''
void testinitonly(int argc, char** argv){
  for(int i = 0; i < argc; i++){
    if(0 == strcmp(argv[i], "initonly")){
      exit(0);
    }
  }
}
''')
        for test_file in POLYBENCH_TEST_C_FILES:
            patch(test_file, r'int main\(int argc, char\*\* argv\)[\r\n]+{', r'\g<0>\n  testinitonly(argc, argv);')


download_if_necessary()
patch_if_necessary()
