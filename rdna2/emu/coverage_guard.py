"""Counted, blocking coverage for difftests: a verdict covers exactly the combinations that ran.

A test declares up front the set of parameter combinations a full claim needs (e.g. all four
shift-window origins). Every run records the combinations it actually exercised in a ledger under
build/coverage/<name>.json. The final verdict prints (required, exercised) and says PASS only when
every required combination has a PASS recorded. A missing combination makes the verdict
INCOMPLETE, never PASS. An incomplete run cannot be reported as a complete one.

The ledger outlives one process because some tests run one combination per invocation
(net_block512_2.py takes MODE from the environment). Each entry is stamped with a fingerprint of the
test script and the kernel code objects it used. An entry whose fingerprint no longer matches is
stale and does not count, so a result from before a code change cannot fill a gap after it.

    cov = Coverage('net_block512_2', required=[{'MODE': m} for m in range(4)],
                   fingerprint_files=[__file__, *modules])
    cov.record({'MODE': 1}, 'PASS', detail='0 mismatches')
    raise SystemExit(cov.verdict())     # 0 only if every required combination has passed

Combinations are dicts. Only the keys named in `required` are compared, so extra context
(geometry, seed) can be stored in `detail` without splitting the key space.
"""
import hashlib, json, os, time
from pathlib import Path

LEDGER_DIR = Path(__file__).resolve().parent.parent / 'build' / 'coverage'


def _key(combo, keys):
    return json.dumps({k: combo[k] for k in sorted(keys)}, sort_keys=True)


def fingerprint(files):
    h = hashlib.sha256()
    for f in sorted(str(Path(x).resolve()) for x in files):
        h.update(f.encode())
        h.update(Path(f).read_bytes() if Path(f).exists() else b'<missing>')
    return h.hexdigest()[:16]


class Coverage:
    def __init__(self, name, required, fingerprint_files, keys=None):
        self.name = name
        self.required = [dict(r) for r in required]
        assert self.required, 'a coverage claim needs at least one required combination'
        self.keys = sorted(keys or self.required[0].keys())
        for r in self.required:
            assert sorted(r.keys()) == self.keys, ('required combinations must share keys', r)
        self.fp = fingerprint(fingerprint_files)
        self.path = LEDGER_DIR / (name + '.json')
        self.ledger = {}
        if self.path.exists():
            try:
                self.ledger = json.loads(self.path.read_text())
            except ValueError:
                self.ledger = {}

    def record(self, combo, status, detail=''):
        """Record one exercised combination. Status is the test's own verdict for it (PASS/FAIL/...)."""
        k = _key(combo, self.keys)
        self.ledger[k] = dict(status=status, fp=self.fp, detail=detail,
                              at=time.strftime('%Y-%m-%d %H:%M:%S'))
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.ledger, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    def verdict(self, out=print):
        """Print the counted verdict. Returns the exit code: 0 only for complete and all-PASS."""
        passed, failed, missing, stale = [], [], [], []
        for r in self.required:
            e = self.ledger.get(_key(r, self.keys))
            if e is None:
                missing.append(r)
            elif e['fp'] != self.fp:
                stale.append(r)
            elif e['status'] == 'PASS':
                passed.append(r)
            else:
                failed.append((r, e['status']))
        n = len(self.required)
        exercised = len(passed) + len(failed)
        out('coverage %s: required %d, exercised %d (current code), passed %d, failed %d, missing %d, stale %d'
            % (self.name, n, exercised, len(passed), len(failed), len(missing), len(stale)))
        for r, st in failed:
            out('  FAILED   %s -> %s' % (r, st))
        for tag, rows, note in (('MISSING', missing, ''), ('STALE  ', stale, ' (recorded under different code; rerun it)')):
            for r in rows[:10]:
                out('  %s  %s%s' % (tag, r, note))
            if len(rows) > 10:
                out('  %s  ... and %d more' % (tag, len(rows) - 10))
        if failed:
            out('VERDICT %s: FAIL' % self.name)
            return 1
        if missing or stale:
            out('VERDICT %s: INCOMPLETE - %d of %d required combinations have a current PASS. '
                'This is NOT a pass for the full claim.' % (self.name, len(passed), n))
            return 2
        out('VERDICT %s: PASS - all %d required combinations exercised and passed' % (self.name, n))
        return 0
