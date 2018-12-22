def read_lines(fname):
    values = []
    with open(fname, 'r') as f:
        values = [x.strip() for x in f if len(x) > 1 and not x.startswith('#')]
    return values

def datetimestr():
    from datetime import datetime as dt
    return dt.now().strftime('%Y%m%d%H%M')

def listdir_image(path):
    import os
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1] in ['.png', '.jpeg', '.jpg', '.bmp', '.gif']]

def listdir_image_r(path):
    result = []
    import os
    entries = [os.path.join(path, e) for e in os.listdir(path)]
    dirs = [e for e in entries if os.path.isdir(e)]
    #result += [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1] in ['.png', '.jpeg', '.jpg', '.bmp', '.gif']]
    result += [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1] in ['.png', '.jpeg', '.jpg', '.bmp']]
    for d in dirs:
        result += listdir_image_r(d)
    return result

def listdir_dir(path):
    import os
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def basename_no_ext(path):
    import os
    bn = os.path.basename(path)
    return os.path.splitext(bn)[0]
