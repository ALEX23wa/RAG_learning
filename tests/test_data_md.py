import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_process import data_md


class MinerUDeviceTests(unittest.TestCase):
    def run_conversion(self, devices, failures, device="auto"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.pdf"
            source.write_bytes(b"test input; worker is mocked")
            calls = []

            def worker(command, **kwargs):
                selected = command[command.index("--device") + 1]
                calls.append((selected, kwargs["env"]))
                error = failures.get(selected)
                if error:
                    kwargs["stdout"].write(error)
                    return subprocess.CompletedProcess(command, 1)
                (root / "markdown.md").write_text("parsed", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(data_md, "mineru_gpu_devices", return_value=devices), \
                 patch.object(data_md.subprocess, "run", side_effect=worker):
                result = data_md.pdf_to_markdown_mineru(source, root, device=device)
            self.assertEqual(result.read_text(), "parsed")
            return calls

    def test_gpu_success_does_not_retry_cpu(self):
        calls = self.run_conversion(["Vulkan0: AMD"], {})
        self.assertEqual([c[0] for c in calls], ["gpu"])

    def test_missing_gpu_selects_cpu(self):
        calls = self.run_conversion([], {})
        self.assertEqual([c[0] for c in calls], ["cpu"])
        self.assertEqual(calls[0][1]["GGML_VK_VISIBLE_DEVICES"], " ")

    def test_gpu_memory_error_retries_in_cpu_process(self):
        calls = self.run_conversion(["Vulkan0: AMD"], {"gpu": "ErrorOutOfDeviceMemory"})
        self.assertEqual([c[0] for c in calls], ["gpu", "cpu"])
        self.assertEqual(calls[1][1]["MINERU_TABLE_DEVICE"], "cpu")

    def test_download_error_is_not_hidden_by_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "Download failed"):
            self.run_conversion(["Vulkan0: AMD"], {"gpu": "Download failed: HTTP 403"})

    def test_cpu_failure_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "CPU failure"):
            self.run_conversion([], {"cpu": "CPU failure"})

    def test_forced_gpu_fails_when_absent(self):
        with self.assertRaisesRegex(RuntimeError, "未检测到可用 GPU"):
            self.run_conversion([], {}, device="gpu")

    def test_forced_cpu_skips_gpu(self):
        calls = self.run_conversion(["Vulkan0: AMD"], {}, device="cpu")
        self.assertEqual([c[0] for c in calls], ["cpu"])

    def test_native_crash_is_retryable(self):
        self.assertTrue(data_md._gpu_failure(3221225477, ""))
        self.assertFalse(data_md._gpu_failure(1, "Invalid PDF"))


if __name__ == "__main__":
    unittest.main()
