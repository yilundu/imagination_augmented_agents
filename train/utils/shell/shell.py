from __future__ import print_function

import os
import shutil


# reference: https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir(path, clean = False):
    if clean:
        rm(path)
    if isinstance(path, list) or isinstance(path, tuple):
        for name in path:
            mkdir(name)
    else:
        if not os.path.exists(path):
            os.makedirs(path)


# reference: https://stackoverflow.com/questions/814167/easiest-way-to-rm-rf-in-python
def rm(path):
    if isinstance(path, list) or isinstance(path, tuple):
        for name in path:
            rm(name)
    else:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)


# reference: https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python
def mv(source, destination):
    shutil.move(source, destination)


# reference: https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python
def cp(source, destination):
    shutil.copy2(source, destination)


def run(command, verbose = False):
    def parse(arguments):
        if isinstance(arguments, str):
            return arguments
        elif isinstance(arguments, list) or isinstance(arguments, tuple):
            concat = ''
            for argument in arguments:
                concat += ' ' + parse(argument)
            return concat[1:]
        else:
            return str(arguments)

    command = parse(command)
    if verbose:
        print('==>', command)
    else:
        command += ' > /dev/null 2>&1'
    return os.system(command)
