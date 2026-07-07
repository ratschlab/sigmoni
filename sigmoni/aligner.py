from __future__ import annotations

import os
import sys
import tempfile
import subprocess as proc
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Iterable

import attrs
import numpy as np
from Bio import SeqIO

from .Bins import HPCBin, SigProcHPCBin
from .run_spumoni import bin_read, MatchingStatisticsParser, count_pmls
from . import utils

try:
    from pyspumoni._core import Index as _SpumoniIndex
    _PYSPUMONI = True
except ImportError:
    _PYSPUMONI = False



def _path_to_species(fname: str) -> str:
    """Extract species name from a shredded reference FASTA filename."""
    fname = os.path.basename(fname)
    for suffix in ('_rc.fasta', '_rc.fa'):
        if fname.endswith(suffix):
            fname = fname[: -len(suffix)] + os.path.splitext(suffix)[1]
            break
    return '_'.join(os.path.splitext(fname)[0].split('_')[:-1])


@attrs.define
class Alignment:
    """Minimal alignment result for readfish compatibility.

    ``ctg`` holds the predicted class (species name or ``pos_class``).
    Numeric fields are unused but required by the readfish interface.
    """
    ctg: str
    ctg_len: int = 0
    r_st: int = 0
    r_en: int = 0
    strand: int = 1
    NM: int = 0
    mapq: int = 0
    is_primary: bool = True


@attrs.define
class Result:
    """Result holder — progressively filled by the basecaller then the aligner."""
    channel: int
    read_id: str
    seq: str
    barcode: Optional[str] = attrs.field(default=None)
    basecall_data: Optional[Any] = attrs.field(default=None)
    alignment_data: Optional[list[Alignment]] = attrs.field(default=None)


class Aligner:
    """Readfish-compatible aligner that classifies reads using Sigmoni.

    Accepted ``**kwargs``:
        ref_prefix (str, required): Path prefix of the SPUMONI index.
        threads (int, default 1): Workers for parallel signal binning; also
            sets ``PARLAY_NUM_THREADS`` for pyspumoni's ``query_batch``.
        multi (bool, default False): Multi-class mode — returns per-species
            classification using PML + document arrays via subprocess.  When
            False, uses the in-process pyspumoni ``Index`` (index loaded once,
            no subprocess).
        threshold (float):
            - Binary mode: minimum ``pres_frac`` from pyspumoni (0–1, default
              0.0 — accept any "Found" result).
            - Multi-class mode: minimum best/second-best PML ratio required
              before assigning a class (default 1.0 — always assign).
        sig_proc (bool, default False): Use ``SigProcHPCBin`` stall filter.
        nbins (int, default 6): Number of uniform signal bins.
        spumoni_path (str, default 'spumoni'): Path to SPUMONI binary (only
            used in multi-class mode).
        complexity (bool, default False): Delta-complexity correction (multi-
            class mode only).
    """

    def __init__(self, debug_log: str | None = None, **kwargs):
        if debug_log:
            if debug_log == 'stdout':
                self.logfile = sys.stdout
            elif debug_log == 'stderr':
                self.logfile = sys.stderr
            else:
                self.logfile = open(debug_log, 'w')
        else:
            self.logfile = None

        self.ref_prefix = kwargs.pop('ref_prefix', None)
        if self.ref_prefix is None:
            print("Must provide a reference index using the '--ref-prefix' parameter")
            sys.exit(1)
        self.ref_prefix = os.path.abspath(self.ref_prefix)

        self.threads = int(kwargs.pop('threads', 1))
        self.multi = kwargs.pop('multi', False)
        self.complexity = kwargs.pop('complexity', False)
        self.spumoni_path = utils.resolve_spumoni_path(kwargs.pop('spumoni_path', 'spumoni'))
        self.threshold = float(kwargs.pop('threshold', 1.0 if self.multi else 0.0))
        nbins = int(kwargs.pop('nbins', 6))
        sig_proc = kwargs.pop('sig_proc', False)

        # Prefer the pore model saved alongside the index (set at build time) so
        # that query binning always matches reference binning exactly.
        BinClass = SigProcHPCBin if sig_proc else HPCBin
        refs_dir = os.path.dirname(self.ref_prefix)
        bins_path = os.path.join(refs_dir, 'poremodel.bins')
        if os.path.exists(bins_path):
            self.bins = BinClass.from_pickle(bins_path)
            print(f"Sigmoni: loaded bin model from {bins_path} (nbins={self.bins.nbins})")
        else:
            self.bins = BinClass(nbins=nbins, poremodel=utils.model_6mer, clip=False)
            print("Sigmoni: poremodel.bins not found, using default R9 6-mer model")

        if self.multi:
            refs_dir = os.path.dirname(self.ref_prefix)
            path = os.path.join(refs_dir, 'filelist.txt')
            self.doc_to_species = {
                int(line.split()[1]): _path_to_species(line.split()[0])
                for line in open(path).read().splitlines()
            }
            self.maxdoc = max(self.doc_to_species.keys())

        if not _PYSPUMONI:
            print("Warning: pyspumoni not found — falling back to subprocess for classification")
            self._index = None
        else:
            os.environ['PARLAY_NUM_THREADS'] = str(self.threads)
            # '-n' disables minimizer digestion (min_digest=false) — sigmoni
            # uses a custom binned alphabet, not DNA minimizers.
            # '-d' enables document array (needed for multi-class).
            args = ['-r', self.ref_prefix, '-P', '-n']
            if self.multi:
                args.append('-d')
            self._index = _SpumoniIndex(args)
            print(f"Sigmoni: loaded pyspumoni index from {self.ref_prefix} "
                  f"({'multi' if self.multi else 'binary'} mode)")

        # Thread pool for parallel signal binning.  Threads share this process,
        # so they never re-import pyspumoni or trigger Parlay re-initialization.
        # NumPy and uncalled4's EventDetector both release the GIL, so true
        # parallelism is achieved without spawning new processes.
        self._executor = ThreadPoolExecutor(max_workers=self.threads) if self.threads > 1 else None

    @property
    def initialised(self) -> bool:
        return True

    def validate(self) -> None:
        if self.multi:
            refs_dir = os.path.dirname(self.ref_prefix)
            path = os.path.join(refs_dir, 'filelist.txt')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Sigmoni: missing {path}")
        elif self._index is not None:
            self._index.validate()
        print("Sigmoni: index OK")

    def describe(self, regions: list, barcodes: dict) -> str:
        if self.multi:
            mode = 'multi-class (subprocess PML)'
        elif self._index is not None:
            mode = 'binary (pyspumoni in-process)'
        else:
            mode = 'binary (subprocess fallback)'
        return (
            f"Sigmoni {mode} — index: {self.ref_prefix}, "
            f"threads: {self.threads}, bins: {self.bins.nbins}"
        )

    def _bin_batch(self, to_bin: list) -> list:
        """Bin a batch of Result objects in parallel; returns (read_id, charseq) pairs."""
        bin_inputs = [
            (r.read_id, np.array(r.basecall_data), self.bins, True)
            for r in to_bin
        ]
        if self._executor is not None:
            return list(self._executor.map(bin_read, bin_inputs))
        return [bin_read(item) for item in bin_inputs]

    def _map_binary_pyspumoni(self, binned_reads, metadata):
        """Binary classification via in-process pyspumoni query_batch."""
        valid = [(rid, seq) for rid, seq in binned_reads if seq]
        null_ids = [rid for rid, seq in binned_reads if not seq]

        if valid:
            charseqs = [seq for _, seq in valid]
            alignments = self._index.query_batch(charseqs)
            for (read_id, _), aln in zip(valid, alignments):
                result = metadata[read_id]
                if aln.ctg != '*' and aln.pres_frac >= self.threshold:
                    result.alignment_data = [Alignment(ctg='pos_class')]
                else:
                    result.alignment_data = []
                yield result

        for read_id in null_ids:
            result = metadata[read_id]
            result.alignment_data = []
            yield result

    def _map_multi_pyspumoni(self, binned_reads, metadata):
        """Multi-class classification via in-process get_pml_batch (no subprocess)."""
        valid = [(rid, seq) for rid, seq in binned_reads if seq]
        null_ids = [rid for rid, seq in binned_reads if not seq]

        if valid:
            charseqs = [seq for _, seq in valid]
            pml_results = self._index.get_pml_batch(charseqs)
            for (read_id, _), (lengths, doc_nums) in zip(valid, pml_results):
                result = metadata[read_id]
                hist = np.zeros(self.maxdoc)
                for length, doc in zip(lengths, doc_nums):
                    if 0 < doc <= self.maxdoc:
                        hist[doc - 1] += length
                if hist.sum() == 0:
                    result.alignment_data = []
                else:
                    best_idx = int(hist.argmax())
                    second = np.partition(hist, -2)[-2] if self.maxdoc > 1 else 0.0
                    ratio = (hist[best_idx] + 1e-10) / (second + 1e-10)
                    if ratio >= self.threshold:
                        result.alignment_data = [Alignment(ctg=self.doc_to_species[best_idx + 1])]
                    else:
                        result.alignment_data = []
                yield result

        for read_id in null_ids:
            result = metadata[read_id]
            result.alignment_data = []
            yield result

    def _map_subprocess(self, binned_reads, metadata):
        """Classify by running spumoni as a subprocess (multi-class or fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readfile = os.path.join(tmpdir, 'reads.fa')
            null_ids = []
            with open(readfile, 'w') as f:
                for read_id, charseq in binned_reads:
                    if charseq:
                        f.write(f'>{read_id}\n{charseq}\n')
                    else:
                        null_ids.append(read_id)

            proc.call([
                self.spumoni_path, 'run',
                '-t', str(self.threads),
                '-r', self.ref_prefix,
                '-p', readfile,
                '-P', '-n', '-d',
            ])

            parser = MatchingStatisticsParser(readfile, docs=True, MS=False)
            read_dict = (
                SeqIO.to_dict(SeqIO.parse(readfile, 'fasta'))
                if self.complexity else None
            )

            for read_id in parser.reads():
                result = metadata[read_id]
                try:
                    hist = count_pmls(
                        read_id, parser, self.maxdoc,
                        string=self.complexity, read_dict=read_dict,
                    )
                    if hist is None:
                        result.alignment_data = []
                    elif self.multi:
                        best_idx = int(hist.argmax())
                        second = np.partition(hist, -2)[-2] if len(hist) > 1 else 0.0
                        ratio = (hist[best_idx] + 1e-10) / (second + 1e-10)
                        if ratio >= self.threshold:
                            result.alignment_data = [Alignment(ctg=self.doc_to_species[best_idx + 1])]
                        else:
                            result.alignment_data = []
                    else:
                        # Subprocess binary fallback: pos_class vs neg_class
                        best_idx = int(hist.argmax())
                        if self.doc_to_species.get(best_idx + 1) == 'pos_class':
                            second = np.partition(hist, -2)[-2] if len(hist) > 1 else 0.0
                            ratio = (hist[best_idx] + 1e-10) / (second + 1e-10)
                            result.alignment_data = (
                                [Alignment(ctg='pos_class')] if ratio >= self.threshold else []
                            )
                        else:
                            result.alignment_data = []
                except (AttributeError, TypeError):
                    result.alignment_data = []
                yield result

            for read_id in null_ids:
                result = metadata[read_id]
                result.alignment_data = []
                yield result

    def map_reads(self, calls: Iterable[Result]) -> Iterable[Result]:
        skipped = []
        to_bin = []
        metadata = {}

        for result in calls:
            metadata[result.read_id] = result
            if result.basecall_data is None:
                skipped.append(result)
            else:
                to_bin.append(result)

        if to_bin:
            binned_reads = self._bin_batch(to_bin)
            if self._index is not None:
                if self.multi:
                    yield from self._map_multi_pyspumoni(binned_reads, metadata)
                else:
                    yield from self._map_binary_pyspumoni(binned_reads, metadata)
            else:
                yield from self._map_subprocess(binned_reads, metadata)

        for result in skipped:
            result.alignment_data = []
            yield result

    def disconnect(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self.logfile and self.logfile not in [sys.stdout, sys.stderr]:
            self.logfile.close()
