import mylib.utils

def read_evaluation_data(classes, rows, cols, data_dir):
    from keras.preprocessing.image import ImageDataGenerator
    import os

    samples = sum([len(mylib.utils.listdir_image(os.path.join(data_dir, c)))
                   for c in classes])

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        directory=data_dir,
        target_size=(cols, rows),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=samples,
        shuffle=False)

    return test_generator.__next__()


def plot_history(histories, fname):
    import matplotlib.pyplot as plt
    def x(key):
        hs = [h.history[key] for h in histories]
        ret = []
        for h in hs:
            ret.extend(h)
        return list(ret)
    # print(history.history.keys())
    plt.figure(figsize=[11.0, 4.5], dpi=100)

    # 精度の履歴をプロット
    if 'acc' in histories[0].history:
        plt.subplot(121)
        #plt.plot(history.history['acc'], "o-")
        #plt.plot(history.history['val_acc'], "o-")
        plt.plot(x('acc'), "o-")
        plt.plot(x('val_acc'), "o-")
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['acc', 'val_acc'], loc='lower right')

    # 損失の履歴をプロット
    plt.subplot(122)
    #plt.plot(history.history['loss'], "o-")
    #plt.plot(history.history['val_loss'], "o-")
    plt.plot(x('loss'), "o-")
    plt.plot(x('val_loss'), "o-")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(fname)
    #plt.show()


def plot_cm(cm, classes, fname):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = plt.axes()
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    plt.title('hahaha')
    plt.colorbar()
    plt.grid(False)
    plt.xlabel('predicted class')
    plt.ylabel('true class')
    plt.grid(False)
    plt.savefig(fname)
    plt.show()


def imaug(img, Debug=False):
    import cv2
    import random
    rows = img.shape[0]
    cols = img.shape[1]
    # rotate
    if random.random() > 0.5:
        degree = random.randrange(0, 360, 90)
        mat = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
        img = cv2.warpAffine(img, mat, (cols, rows))
    if random.random() > 0.5:
        degree = random.uniform(-15, 15)
        #print(degree)
        rows = img.shape[0]
        cols = img.shape[1]
        mat = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
        img = cv2.warpAffine(img, mat, (cols, rows))

    ## crop
    #rate = random.uniform(0.8, 1.0)
    #rows = img.shape[0]
    #cols = img.shape[1]
    #w = int(cols * rate)
    #h = int(rows * rate)
    #x = int(random.uniform(0, cols - w))
    #y = int(random.uniform(0, rows - h))
    #print('crop', (x, y), w, 'x', h)
    #img = img[y:y+h, x:x+w, :]

    #img = cv2.resize(img, (224, 224))

    # flip (horizontal)
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if random.random() > 0.5:
        img = cv2.flip(img, 0)

    #Debug = True
    if Debug:
        cv2.imshow('imaug debug', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def read_image(fname, area=None, size=None, converter=None):
    import cv2
    img = cv2.imread(fname)
    if img is None:
        #print(fname)
        fname_cp932 = fname.encode('utf-8').decode('cp932')
        img = cv2.imread(fname_cp932)
        if img is None:
            raise RuntimeError('cv2.imread(failed): ' + fname)
    if area is not None:
        x, y, w, h = area
        img = img[y:y+h, x:x+w]
    if converter is not None:
        img = converter(img)
    if size is not None:
        img = cv2.resize(img, size)
    return img

