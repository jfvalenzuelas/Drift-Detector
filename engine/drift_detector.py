import sys

path, language, windows, keywords = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def print_arguments(path, language, windows, keywords):
    parameters = {
        "path": path,
        "language": language,
        "windows": windows,
        "keywords": keywords
    }
    return parameters

print(print_arguments(path, language, windows, keywords))
sys.stdout.flush()