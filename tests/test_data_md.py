import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_process import data_md


class MinerUDeviceTests(unittest.TestCase):
    def run_conversion(self, devices, failures, device="auto"):
        """在临时目录模拟 MinerU 转换，记录设备尝试顺序与环境。

        Args:
            devices: list[str]：模拟探测到的 GPU 描述。
            failures: dict[str, str]：设备名到模拟失败日志的映射；没有条目表示该设备成功。
            device: str：传给生产转换函数的设备策略，默认 auto。

        Returns:
            list[tuple[str, dict]]：每次模拟子进程调用的设备名和环境变量。

        Notes:
            mock 真实子进程和设备探测，不加载模型；断言生成了预期 Markdown。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.pdf"
            source.write_bytes(b"test input; worker is mocked")
            calls = []

            def worker(command, **kwargs):
                """模拟指定设备的 MinerU 工作进程。

                Args:
                    command: list[str]：生产函数构造的命令行，含 --device 参数。
                    kwargs: dict：subprocess.run 的关键字参数，读取 env/stdout 来记录环境和模拟日志。

                Returns:
                    subprocess.CompletedProcess：失败时退出码 1；成功时写模拟 Markdown 并返回 0。
                """
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
        """验证 GPU 成功时仅运行一次 GPU 解析，不额外调用 CPU。

        Args:
            无显式输入参数。

        Returns:
            None：使用断言判断测试结果；不满足条件时测试失败。
        """
        calls = self.run_conversion(["Vulkan0: AMD"], {})
        self.assertEqual([c[0] for c in calls], ["gpu"])

    def test_missing_gpu_selects_cpu(self):
        """验证未发现 GPU 时选择 CPU 并隐藏 Vulkan 设备。

        Args:
            无显式输入参数。

        Returns:
            None：使用断言判断测试结果；不满足条件时测试失败。
        """
        calls = self.run_conversion([], {})
        self.assertEqual([c[0] for c in calls], ["cpu"])
        self.assertEqual(calls[0][1]["GGML_VK_VISIBLE_DEVICES"], " ")

    def test_gpu_memory_error_retries_in_cpu_process(self):
        """验证 GPU 显存不足后在独立 CPU 环境中重试。

        Args:
            无显式输入参数。

        Returns:
            None：使用断言判断测试结果；不满足条件时测试失败。
        """
        calls = self.run_conversion(["Vulkan0: AMD"], {"gpu": "ErrorOutOfDeviceMemory"})
        self.assertEqual([c[0] for c in calls], ["gpu", "cpu"])
        self.assertEqual(calls[1][1]["MINERU_TABLE_DEVICE"], "cpu")

    def test_download_error_is_not_hidden_by_fallback(self):
        """验证模型下载错误直接上报，不错误回退为 CPU 解析。

        Args:
            无显式输入参数。

        Returns:
            None：使用断言判断测试结果；不满足条件时测试失败。
        """
        with self.assertRaisesRegex(RuntimeError, "Download failed"):
            self.run_conversion(["Vulkan0: AMD"], {"gpu": "Download failed: HTTP 403"})

    def test_cpu_failure_is_reported(self):
        """验证 CPU 解析失败保留并抛出原错误信息。

        Args:
            无显式输入参数。

        Returns:
            None：使用断言判断测试结果；不满足条件时测试失败。
        """
        with self.assertRaisesRegex(RuntimeError, "CPU failure"):
            self.run_conversion([], {"cpu": "CPU failure"})

    def test_forced_gpu_fails_when_absent(self):
        """验证强制 GPU 而无可用设备时明确失败。

        Args:
            无显式输入参数。

        Returns:
            None：使用断言判断测试结果；不满足条件时测试失败。
        """
        with self.assertRaisesRegex(RuntimeError, "未检测到可用 GPU"):
            self.run_conversion([], {}, device="gpu")

    def test_forced_cpu_skips_gpu(self):
        """验证强制 CPU 时跳过 GPU 探测和 GPU 解析。

        Args:
            无显式输入参数。

        Returns:
            None：使用断言判断测试结果；不满足条件时测试失败。
        """
        calls = self.run_conversion(["Vulkan0: AMD"], {}, device="cpu")
        self.assertEqual([c[0] for c in calls], ["cpu"])

    def test_native_crash_is_retryable(self):
        """验证已知 native 崩溃码可重试，普通非法 PDF 错误不可重试。

        Args:
            无显式输入参数。

        Returns:
            None：使用断言判断测试结果；不满足条件时测试失败。
        """
        self.assertTrue(data_md._gpu_failure(3221225477, ""))
        self.assertFalse(data_md._gpu_failure(1, "Invalid PDF"))


if __name__ == "__main__":
    unittest.main()
