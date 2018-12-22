import codecs
import glob
import json
import os
import shutil

import numpy as np

import mylib.image


def load_default_config():
    base_dir = os.path.join(mylib.image.rootdir(), 'models', 'attribute')
    config_path = os.path.join(base_dir, 'config.json')
    with codecs.open(config_path, 'r', 'utf_8') as f:
        config = json.load(f)
    dataset_fname = os.path.join(base_dir, config['dataset'])
    with codecs.open(dataset_fname, 'r', 'utf_8') as f:
        dataset = json.load(f)
    return dataset['tags'], os.path.join(base_dir, config['weights'])


def decode_predictions(preds, tags, threashold=0.5):
    #return [ for idx, tag in enumerate(preds)]
    result = []
    for idx, tag in enumerate(tags):
        if preds[idx][0][1] > threashold:
            result.append((tags[idx][0], preds[idx][0][1]))
    return result


def predict(img_path, tags, weights):
    from keras.preprocessing import image
    from mylib.image import xception_ft
    import numpy as np

    if not isinstance(img_path, list):
        img_path = [img_path]

    classes = [2] * len(tags)
    m = xception_ft.Xception_ft(classes, weights=weights)

    result = []
    for fpath in img_path:
        img = image.load_img(fpath, target_size=(299, 299))
        #print(img)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.
        
        preds = m.predict(x)
        #print(preds)
        result.append(decode_predictions(preds, tags))
        #result.append(preds)

    return result


def run_predict(args):
    print('run_predict', args)

    if args.dataset_json is None or args.weights is None:
        default_tags, default_weights = load_default_config()

    if args.weights is None:
        weights = default_weights
    else:
        weights = args.weights

    if args.dataset_json is None:
        tags = default_tags
    else:
        with codecs.open(args.dataset_json, 'r', 'utf_8') as f:
            info = json.load(f)
        tags = info['tags']



    paths = []
    for path in args.path:
        paths += glob.glob(path)
    print(paths)

    preds = predict(paths, tags, weights)
    for path, pred in zip(paths, preds):
        print(path, pred, sep=': ')


def train(args):
    print('train', args)

    from   mylib.image.datagen3 import DataGen3, read_image_from_items
    import mylib.utils
    from   mylib.image import xception_ft

    output_dir = os.path.join(args.output_dir, mylib.utils.datetimestr())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_json = os.path.join(args.dataset_dir, 'dataset.json')
    with codecs.open(dataset_json, 'r', 'utf_8') as f:
        info = json.load(f)
    output_names = ['output{}'.format(i) for i in range(len(info['tags']))]
    shutil.copy2(dataset_json, output_dir)

    image_dir = os.path.join(args.dataset_dir, 'images')
    train_generator = DataGen3(info['train'], info['tags'],
                               output_names=output_names,
                               size=(299, 299),
                               batch_size=32,
                               image_dir=image_dir)
    #print(len(gen))
    if args.samples_per_epoch is not None:
        steps_per_epoch = args.samples_per_epoch // args.batch_size
    else:
        steps_per_epoch = len(train_generator) // args.batch_size

    test_fname = os.path.join(args.dataset_dir, 'test.npz')
    with np.load(test_fname) as data:
        d = dict(data)
        x_test = d.pop('x')
        print(x_test.shape)
        y_test = d
        validation_data = (x_test, y_test)
    #validation_data = read_image_from_items(info['test'],
    #                                        output_names,
    #                                        size=(299, 299),
    #                                        image_dir=args.image_dir)
    #print(validation_data[0])

    #for i in range(len(info['tags'])):
    #    losses = {'output{}'.format(idx): 'categorical_crossentropy'
    #              for idx in range(len(classes))}
    losses = {'output{}'.format(idx): 'categorical_crossentropy'
              for idx in range(len(info['tags']))}
    print(losses)

    #model = Xception_ft([2, 2], weights=None).summary()
    #print([2] * len(info['tags']))
    xception_ft.train(train_generator, validation_data,
                      classes=[2] * len(info['tags']),
                      losses=losses,
                      model_name='attribute',
                      output_dir=output_dir,
                      steps_per_epoch=steps_per_epoch,
                      patience=args.patience)


def evaluate(args):
    #print('evaluate', args)

    from mylib.image import cartoon
    result = cartoon.evaluate_generator(args.path, 1)
    print(result)


def printmodel(args):
    from mylib.image.xception_ft import Xception_ft
    #Xception_ft(2, weights=None).summary()
    #Xception_ft([2], weights=None).summary()
    Xception_ft([2, 2], weights=None).summary()


def options():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='proccess images')
    subparsers = parser.add_subparsers()

    predict_parser = subparsers.add_parser('predict', help='predict image')
    predict_parser.add_argument('--dataset-json', type=str, default=None)
    predict_parser.add_argument('--weights', type=str, default=None)
    predict_parser.add_argument('path', type=str, default=None, nargs='+')
    predict_parser.set_defaults(func=run_predict)

    train_parser = subparsers.add_parser('train', help='train model')
    train_parser.add_argument('--dataset-dir', type=str, required=True)
    #train_parser.add_argument('--image-dir', type=str, default='images')
    train_parser.add_argument('--output-dir', type=str, default='attribute')
    train_parser.add_argument('--samples-per-epoch', type=int, default=None)
    train_parser.add_argument('--batch-size', type=int, default=50)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--weights', type=str, default=None)
    train_parser.add_argument('--patience', type=int, default=10)

    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser('evaluate', help='evaluate images')
    evaluate_parser.add_argument('path', help='path to the validation data directory')
    evaluate_parser.set_defaults(func=evaluate)

    evaluate_parser = subparsers.add_parser('printmodel', help='print model information')
    evaluate_parser.set_defaults(func=printmodel)

    return parser.parse_args()


def main():
    args = options()
    print('options:', args)
    args.func(args)


if __name__ == '__main__':
    main()
