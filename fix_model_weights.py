import os
import sys
import tarfile

import torch


def main(infile):
    with tarfile.open(infile) as intar:
        fp = intar.extractfile('weights.th')
        state_dict = torch.load(fp)

    print('total keys:', len(state_dict))
    bert_weights = {
        k[len('model.'):]: v for k, v in state_dict.items()
        if k.startswith('model.')
    }

    print('bert keys:', len(bert_weights))
    clf_weights = {
        k[len('classifier.'):]: v for k, v in state_dict.items()
        if k.startswith('classifier.')
    }

    for fname, weights_dict in [('bert.th', bert_weights), ('clf.th', clf_weights)]:
        print(f'saving {fname}')
        torch.save(weights_dict, os.path.join(os.path.dirname(infile), fname))
    print('done')


if __name__ == '__main__':
    main(sys.argv[1])
