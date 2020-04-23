import glob
import json
import os
import random
import sys

patterns = {
    'Hypernymy_S*_1_A.txt': 'hp_s_a_train.jsonl',
    'Hypernymy_S*_2_A.txt': 'hp_s_a_test.jsonl',
    'Hypernymy_S*_2_B.txt': 'hp_s_b_test.jsonl',
    'Hypernymy_S*_2_IB.txt': 'hp_s_ib_test.jsonl',
}


def main(indir, val_ratio=0.15):
    os.makedirs('hypernymy_dataset', exist_ok=True)

    for pattern, outfname in patterns.items():
        infiles = list(glob.glob(os.path.join(indir, pattern)))
        samples = [s for f in infiles for s in read_file(f)]

        if outfname == 'hp_s_a_train.jsonl':
            random.seed(1337)
            random.shuffle(samples)
            split_pos = int(len(samples) * (1 - val_ratio))
            train_samples = samples[:split_pos]
            valid_samples = samples[split_pos:]
            write_file(train_samples, outfname, infiles)
            write_file(valid_samples, outfname.replace('train', 'valid'), infiles)
        else:
            write_file(samples, outfname, infiles)


def read_file(fname):
    with open(fname) as fp:
        lines = [l.strip() for l in fp if l.strip() and not l.startswith('#')]
    samples = [
        {'sentence1': s1, 'sentence2': s2, 'gold_label': l}
        for s1, s2, l in chunkwise(lines, size=3)
    ]

    # validate dataset
    for s in samples:
        if not (
            s['gold_label'] in ('contradiction', 'entailment', 'neutral')
            and len(s['sentence1']) > 3
            and len(s['sentence2']) > 3
        ):
            raise Exception(f'invalid sample - {s}')
    return samples


def chunkwise(t, size):
    """variable length chunks from iterable"""
    it = iter(t)
    return zip(*[it]*size)


def write_file(lines, fname, infiles=(), outdir='hypernymy_dataset'):
    print(f'saving {fname} from {infiles} with {len(lines)} samples')

    filename = os.path.join(outdir, fname)
    with open(filename, 'w') as fp:
        for l in lines:
            fp.write(json.dumps(l))
            fp.write('\n')


if __name__ == '__main__':
    main(indir=sys.argv[1])
