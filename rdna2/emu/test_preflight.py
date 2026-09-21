"""Protocol negative controls: mocked process, no GPU access."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import difftest_var as D


class PreflightTests(unittest.TestCase):
    def exercise(self, ready, callback):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'test').mkdir()
            (root / 'test' / 'kernarg.bin.rebased').write_bytes(b'actual')
            p = Mock()
            p.stdout = io.StringIO(ready)
            p.poll.return_value = None
            p.returncode = 0
            p.communicate.return_value = ('finished', '')
            self.process = p
            with patch.object(D, 'OUT', root), patch.object(D.subprocess, 'Popen', return_value=p):
                return D.gpu(Path('module'), 'symbol', 'test', b'input', np.zeros(8, np.uint8), (1, 1), callback)

    def test_success_validates_before_go(self):
        def validate(base, args):
            self.assertEqual(base, 0x404010000)
            self.assertEqual(args, b'actual')
            self.process.communicate.assert_not_called()
        self.exercise('READY arena_dev=0x404010000\n', validate)
        self.process.communicate.assert_called_once_with('GO\n', timeout=60)

    def test_rejected_fixture_never_sends_go(self):
        def reject(base, args):
            raise ValueError('invalid fixture')
        with self.assertRaises(ValueError):
            self.exercise('READY arena_dev=0x404010000\n', reject)
        self.process.communicate.assert_not_called()
        self.process.kill.assert_called_once()

    def test_bad_handshake_never_validates_or_launches(self):
        callback = Mock()
        with self.assertRaises(RuntimeError):
            self.exercise('HIP error\n', callback)
        callback.assert_not_called()
        self.process.communicate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
