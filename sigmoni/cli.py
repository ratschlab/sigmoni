"""Command-line entry points for sigmoni.

Registered as console_scripts in pyproject.toml:
    sigmoni-index    -> sigmoni.cli:index
    sigmoni-classify -> sigmoni.cli:classify
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import subprocess as proc
from collections import namedtuple

import numpy as np
from Bio import SeqIO
from sklearn.metrics import precision_recall_curve
from tqdm.auto import tqdm
from uncalled4.read_index import Fast5Reader

from . import utils
from . import run_spumoni as sig
from .Bins import HPCBin, SigProcHPCBin
from .aligner import _path_to_species

_read = namedtuple('read', ['id', 'signal'])


_resolve_spumoni_path = utils.resolve_spumoni_path


# ---------------------------------------------------------------------------
# sigmoni-index
# ---------------------------------------------------------------------------

def _index_parse_args():
    parser = argparse.ArgumentParser(
        description="Build a SPUMONI index over reference FASTA files for sigmoni classification"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-pl', dest='pos_filelist', help='file listing positive reference FASTAs')
    group.add_argument('-p', dest='pos_filelist', nargs='+', help='positive reference FASTA(s)')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-nl', dest='null_filelist', default=[], help='file listing negative reference FASTAs')
    group.add_argument('-n', dest='null_filelist', default=[], nargs='+', help='negative reference FASTA(s)')

    parser.add_argument('--poremodel', dest='poremodel', default=None,
                        help='path to pore model TSV file (default: built-in R9 6-mer model)')
    parser.add_argument('-b', '--nbins', dest='nbins', default=6, type=int,
                        help='number of bins to discretize signal (default: 6)')
    parser.add_argument('--shred', dest='shred_size', default=int(1e5), type=int,
                        help='shred window size in bp (default: 100000; 0 = no shredding)')
    parser.add_argument('--no-rev-comp', action='store_false', default=True, dest='rev_comp',
                        help='skip reverse-complement reference shreds')
    parser.add_argument('--no-build', action='store_true', default=False, dest='no_build',
                        help='bin references without building the SPUMONI index')
    parser.add_argument('--spumoni-path', dest='spumoni_path', default='spumoni',
                        help='path to spumoni binary (default: spumoni on PATH)')
    parser.add_argument('-o', default='./', dest='output_path', help='output directory (default: ./)')
    parser.add_argument('--ref-prefix', dest='ref_prefix', default='ref',
                        help='prefix for index output files (default: ref)')
    return parser.parse_args()


def _index_format_args(args):
    if isinstance(args.pos_filelist, str):
        args.pos_filelist = list(map(os.path.abspath, open(args.pos_filelist).read().splitlines()))
    if isinstance(args.null_filelist, str):
        args.null_filelist = list(map(os.path.abspath, open(args.null_filelist).read().splitlines()))
    args.output_path = os.path.abspath(args.output_path)
    args.spumoni_path = _resolve_spumoni_path(args.spumoni_path)
    poremodel = args.poremodel if args.poremodel else utils.model_6mer
    args.bins = HPCBin(nbins=args.nbins, poremodel=poremodel, clip=False)


def _bin_reference(args, files):
    outdir = os.path.join(args.output_path, 'refs/')
    os.makedirs(outdir, exist_ok=True)
    docs = []
    for ref in tqdm(files):
        out_fname = os.path.join(outdir, os.path.basename(ref))
        docs += sig.write_shredded_ref(ref, args.bins, out_fname, header=True,
                                       revcomp=args.rev_comp, shred_size=args.shred_size)
    if args.shred_size == 0:
        return docs
    sortorder = []
    for fname in docs:
        base = os.path.basename(fname)
        rc = base.endswith('_rc.fasta') or base.endswith('_rc.fa')
        if rc:
            base = base.replace('_rc', '')
        sortorder.append((
            1 if rc else 0,
            base.split('_')[0],
            int(os.path.splitext(base)[0].split('_')[-1]) * (-1 if rc else 1),
        ))
    return [doc for _, doc in sorted(zip(sortorder, docs))]


def _build_reference(args):
    pos_docs = _bin_reference(args, args.pos_filelist)
    pos_filelist = os.path.join(args.output_path, 'refs', 'pos_filelist.txt')
    if args.null_filelist:
        null_docs = _bin_reference(args, args.null_filelist)
        docs = pos_docs + null_docs
    else:
        docs = pos_docs

    with open(pos_filelist, 'w') as f:
        f.write('\n'.join('%s %d' % (r, i) for i, r in enumerate(pos_docs, 1)))
    if args.null_filelist:
        null_filelist = os.path.join(args.output_path, 'refs', 'null_filelist.txt')
        with open(null_filelist, 'w') as f:
            f.write('\n'.join('%s %d' % (r, i)
                              for i, r in enumerate(null_docs, len(pos_docs) + 1)))
    filelist = os.path.join(args.output_path, 'refs', 'filelist.txt')
    with open(filelist, 'w') as f:
        f.write('\n'.join('%s %d' % (r, i) for i, r in enumerate(docs, 1)))

    args.bins.save_bins(os.path.join(args.output_path, 'refs', 'poremodel.bins'))
    if args.no_build:
        sys.exit(0)
    proc.call([args.spumoni_path, 'build', '-i', filelist,
               '-o', os.path.join(args.output_path, 'refs', args.ref_prefix),
               '-P', '-n', '-d', '--no-rev-comp', '-p', '110'])


def index():
    """Entry point for ``sigmoni-index``."""
    args = _index_parse_args()
    _index_format_args(args)
    _build_reference(args)


# ---------------------------------------------------------------------------
# sigmoni-classify
# ---------------------------------------------------------------------------

def _classify_parse_args():
    parser = argparse.ArgumentParser(
        description="Classify nanopore reads directly from raw signal using a sigmoni index"
    )
    parser.add_argument('-i', dest='fast5', required=True,
                        help='path to input fast5 directory')
    parser.add_argument('-r', '--ref-prefix', dest='ref_prefix', required=True,
                        help='reference index prefix (from sigmoni-index)')
    parser.add_argument('-b', '--nbins', dest='nbins', default=6, type=int,
                        help='number of signal bins (default: 6)')
    parser.add_argument('--spumoni-path', dest='spumoni_path', default='spumoni',
                        help='path to spumoni binary (default: spumoni on PATH)')
    parser.add_argument('-o', default='./', dest='output_path', help='output directory (default: ./)')
    parser.add_argument('-t', default=1, dest='threads', type=int, help='number of threads (default: 1)')
    parser.add_argument('--sp', '--sig-proc', action='store_true', dest='sig_proc', default=False,
                        help='filter sequencing stalls before binning')
    parser.add_argument('--read-prefix', dest='read_prefix', default='reads',
                        help='prefix for output read files (default: reads)')
    parser.add_argument('--max-chunks', dest='max_chunk', default=0, type=int,
                        help='maximum number of signal chunks per read (0 = unlimited)')
    parser.add_argument('--no-classify', action='store_true', default=False, dest='no_classify',
                        help='bin and run SPUMONI but skip classification')
    parser.add_argument('--reclassify', action='store_true', default=False, dest='reclassify',
                        help='skip signal processing and re-classify from existing .pseudo_lengths')
    parser.add_argument('-a', dest='annotations', default=None,
                        help='annotation TSV (read_id<tab>pos_class|neg_class) for threshold tuning')
    parser.add_argument('--thresh', dest='threshold', default=1.0, type=float,
                        help='spike-ratio threshold for binary classification (default: 1.0)')
    parser.add_argument('--multi', dest='multi', action='store_true', default=False,
                        help='multi-class classification mode (default: binary)')
    parser.add_argument('--complexity', dest='complexity', action='store_true', default=False,
                        help='enable delta-complexity correction')
    return parser.parse_args()


def _classify_format_args(args):
    args.output_path = os.path.abspath(args.output_path)
    args.ref_prefix = os.path.abspath(args.ref_prefix)
    args.spumoni_path = _resolve_spumoni_path(args.spumoni_path)
    if args.sig_proc:
        print('using signal processing bins')
        args.bins = SigProcHPCBin(nbins=args.nbins, poremodel=utils.model_6mer, clip=False)
    else:
        args.bins = HPCBin(nbins=args.nbins, poremodel=utils.model_6mer, clip=False)
    if args.annotations:
        args.annotations = {
            line.split()[0]: 1 if line.split()[1] == 'pos_class' else 0
            for line in open(args.annotations).read().splitlines()
        }


def _signal_generator(args, signal):
    if args.max_chunk == 0:
        yield from signal
    else:
        for s in signal:
            yield _read(s.id, np.array(s.signal)[:4000 * args.max_chunk])


def _query_reads(args):
    fast5s = []
    for f in os.listdir(args.fast5):
        if f.endswith('.fast5') or f.endswith('.f5'):
            fast5s.append(_signal_generator(args, Fast5Reader(os.path.join(args.fast5, f))))
    seq_signal = itertools.chain(*fast5s)
    readfile = os.path.join(args.output_path, args.read_prefix + '.fa')
    if not os.path.exists(readfile):
        sig.write_read_parallel(seq_signal, args.bins, evdt=utils.SIGMAP_EVDT,
                                fname=readfile, threads=args.threads)
    else:
        print('Using binned query found in: %s' % readfile)
    proc.call([args.spumoni_path, 'run', '-t', str(args.threads),
               '-r', args.ref_prefix, '-p', readfile, '-P', '-n', '-d'])


def _write_classifications(args, preds, suffix):
    outfile = os.path.join(args.output_path, args.read_prefix + suffix + '.report')
    with open(outfile, 'w') as f:
        f.write('read_id\tclass\n')
        for r, p in preds.items():
            f.write('%s\t%s\n' % (r, p))


def _multi_classify(args, parser, doc_to_species, read_dict=None):
    maxdoc = max(doc_to_species.keys())
    preds = {r: sig.best_shred(r, parser, doc_to_species, maxdoc,
                               string=args.complexity, read_dict=read_dict)
             for r in parser.reads()}
    _write_classifications(args, preds, '_multi')


def _binary_classify(args, parser, doc_to_species, read_dict=None):
    maxdoc = max(doc_to_species.keys())
    ratios = {r: sig.spike_test(r, parser, doc_to_species, maxdoc,
                                string=args.complexity, read_dict=read_dict)
              for r in tqdm(parser.reads())}
    if args.annotations:
        p, r, threshold = precision_recall_curve(
            [args.annotations[r] for r in parser.reads()],
            [ratios[r] for r in parser.reads()],
        )
        filt = np.where(p + r != 0)[0]
        p, r = p[filt], r[filt]
        f1s = 2 * (p * r) / (p + r)
        best = f1s.argmax()
        print('Precision, Recall, F1: ', p[best], r[best], f1s[best])
        print('Threshold: ', threshold[best])
    else:
        preds = {r: 'pos_class' if ratios[r] >= args.threshold else 'neg_class'
                 for r in parser.reads()}
        _write_classifications(args, preds, '_binary')


def _classify_reads(args):
    if args.multi:
        path = os.path.join(os.path.dirname(args.ref_prefix), 'filelist.txt')
        doc_to_species = {int(x.split()[1]): _path_to_species(x.split()[0])
                         for x in open(path).read().splitlines()}
    else:
        pos_path = os.path.join(os.path.dirname(args.ref_prefix), 'pos_filelist.txt')
        doc_to_species = {int(x.split()[1]): 'pos_class'
                         for x in open(pos_path).read().splitlines()}
        null_path = os.path.join(os.path.dirname(args.ref_prefix), 'null_filelist.txt')
        doc_to_species |= {int(x.split()[1]): 'neg_class'
                          for x in open(null_path).read().splitlines()}

    readfile = os.path.join(args.output_path, args.read_prefix + '.fa')
    read_dict = SeqIO.to_dict(SeqIO.parse(readfile, 'fasta')) if args.complexity else None
    parser = sig.MatchingStatisticsParser(readfile, docs=True, MS=False)
    if args.multi:
        _multi_classify(args, parser, doc_to_species, read_dict=read_dict)
    else:
        _binary_classify(args, parser, doc_to_species, read_dict=read_dict)


def classify():
    """Entry point for ``sigmoni-classify``."""
    print('Running command: ' + ' '.join(sys.argv))
    args = _classify_parse_args()
    _classify_format_args(args)
    if args.reclassify:
        print('Classifying reads')
        _classify_reads(args)
        return
    print('Querying reads')
    _query_reads(args)
    if not args.no_classify:
        print('Classifying reads')
        _classify_reads(args)
