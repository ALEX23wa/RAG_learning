import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

from data_process.create_block import VisualizedBGEEmbeddings, create_block, split_markdown


class KeywordEmbeddings(Embeddings):
    """可预测的离线向量，用于验证真实向量检索和父块回取。"""

    def __init__(self):
        self.embedded_texts = []

    def embed_documents(self, texts):
        self.embedded_texts.extend(texts)
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.count("苹果")), float(text.count("汽车")), 0.01]


class MarkdownBlockTests(unittest.TestCase):
    def test_headers_and_fenced_code(self):
        text = "前言\n\n# 文档\n介绍\n\n## 方法\n```python\n# 代码注释\n```\n正文\n\n## 结果\n成功"
        parents, children = split_markdown(text)
        method = next(p for p in parents if p.metadata.get("content_type") == "code")
        self.assertEqual(method.metadata["Header 1"], "文档")
        self.assertIn("# 代码注释", method.page_content)
        self.assertTrue(any("前言" in p.page_content for p in parents))
        self.assertFalse(any(p.metadata.get("Header 1") == "代码注释" for p in parents))
        self.assertNotIn("成功", method.page_content)
        self.assertEqual(len(parents), len(children))

    def test_size_limits_relationship_and_no_content_loss(self):
        text = "正文" * 350
        parents, children = split_markdown(
            text, chunk_size=70, chunk_overlap=10, parent_chunk_size=180,
        )
        self.assertGreater(len(parents), 1)
        self.assertEqual("".join(p.page_content for p in parents), text)
        parent_map = {p.id: p for p in parents}
        self.assertTrue(all(len(p.page_content) <= 180 for p in parents))
        self.assertTrue(all(len(c.page_content) <= 70 for c in children))
        self.assertEqual(len({c.id for c in children}), len(children))
        for child in children:
            self.assertIn(child.page_content, parent_map[child.metadata["parent_id"]].page_content)
            self.assertEqual(child.metadata["chunk_type"], "child")
        self.assertTrue(all(p.metadata["chunk_type"] == "parent" for p in parents))

    def test_overlap_for_unbroken_text(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        _, children = split_markdown(text, chunk_size=10, chunk_overlap=3)
        self.assertEqual(children[0].page_content[-3:], children[1].page_content[:3])
        self.assertEqual(children[0].page_content + "".join(c.page_content[3:] for c in children[1:]), text)

    def test_ids_are_repeatable_and_source_specific(self):
        first = split_markdown("# 标题\n正文", source="a.md")
        repeated = split_markdown("# 标题\n正文", source="a.md")
        other = split_markdown("# 标题\n正文", source="b.md")
        self.assertEqual(first, repeated)
        self.assertNotEqual(first[0][0].id, other[0][0].id)

    def test_empty_and_invalid_parameters(self):
        self.assertEqual(split_markdown(" \n\t"), ([], []))
        for kwargs in (
            {"chunk_size": 0}, {"chunk_overlap": -1},
            {"chunk_size": 20, "chunk_overlap": 20}, {"parent_chunk_size": 500},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                split_markdown("正文", **kwargs)

    def test_file_index_and_parent_retrieval(self):
        embeddings = KeywordEmbeddings()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文.md"
            path.write_text("# 水果\n" + "苹果营养丰富。" * 25 + "\n# 交通\n" + "汽车方便出行。" * 25, encoding="utf-8-sig")
            result = create_block(path, chunk_size=35, chunk_overlap=5, embeddings=embeddings)
            self.assertEqual(embeddings.embedded_texts, [c.page_content for c in result.child_documents])
            self.assertTrue(all(c.metadata["source"] == str(path.resolve()) for c in result.child_documents))
            parents = result.search_parents("苹果", k=3)
            self.assertEqual(len(parents), 1)
            self.assertEqual(parents[0].metadata["Header 1"], "水果")
            self.assertGreater(len(parents[0].page_content), 35)
            # 同一父块多次命中时去重，且保留首次命中的顺序。
            fruit = next(c for c in result.child_documents if c.metadata["Header 1"] == "水果")
            car = next(c for c in result.child_documents if c.metadata["Header 1"] == "交通")
            with patch.object(result.vector_store, "similarity_search", return_value=[car, fruit, car]):
                self.assertEqual([p.metadata["Header 1"] for p in result.search_parents("查询")], ["交通", "水果"])

    def test_empty_file_skips_embedding(self):
        embeddings = KeywordEmbeddings()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.md"
            path.write_text("", encoding="utf-8")
            result = create_block(path, embeddings=embeddings)
            self.assertEqual(result.parent_documents, {})
            self.assertEqual(result.search_parents("任意查询"), [])
            self.assertEqual(embeddings.embedded_texts, [])
            with self.assertRaises(ValueError):
                result.search_parents("查询", k=0)

    def test_short_structures_are_preserved_verbatim(self):
        code = '```python\ndef run():\n    print("a | b")\n\n    return 1\n```'
        table = "| 名称 | 数量 |\n| :--- | ---: |\n| 苹果 | 3 |"
        parents, children = split_markdown("# 示例\n\n" + code + "\n\n" + table + "\n\n结束")
        for documents in (parents, children):
            self.assertEqual(next(d.page_content for d in documents if d.metadata["content_type"] == "code"), code)
            self.assertEqual(next(d.page_content for d in documents if d.metadata["content_type"] == "table"), table)
            self.assertTrue(all(d.metadata["Header 1"] == "示例" for d in documents))

    def test_long_code_preserves_every_line_and_indentation(self):
        rows = [f"    value_{i} = {i}" for i in range(35)]
        parents, children = split_markdown("```python\n" + "\n".join(rows) + "\n```",
            chunk_size=80, chunk_overlap=15, parent_chunk_size=160)
        self.assertGreater(len(parents), 1)
        self.assertGreater(len(children), len(parents))
        parent_ids = {p.id for p in parents}
        for documents, size in ((parents, 160), (children, 80)):
            recovered = []
            for document in documents:
                lines = document.page_content.splitlines()
                self.assertEqual(lines[0], "```python")
                self.assertEqual(lines[-1], "```")
                self.assertLessEqual(len(document.page_content), size)
                self.assertIn(document.metadata["parent_id"], parent_ids)
                recovered.extend(lines[1:-1])
            self.assertEqual(recovered, rows)

    def test_long_table_repeats_headers_without_losing_or_duplicating_rows(self):
        header = ["名称 | 值", ":--- | ---:"]
        rows = [f"项目{i} | {i}" for i in range(30)]
        parents, children = split_markdown("\n".join(header + rows),
            chunk_size=65, chunk_overlap=10, parent_chunk_size=125)
        self.assertGreater(len(parents), 1)
        for documents, size in ((parents, 125), (children, 65)):
            recovered = []
            for document in documents:
                lines = document.page_content.splitlines()
                self.assertEqual(lines[:2], header)
                self.assertLessEqual(len(document.page_content), size)
                self.assertEqual(document.metadata["content_type"], "table")
                recovered.extend(lines[2:])
            self.assertEqual(recovered, rows)

    def test_unclosed_and_nested_fences(self):
        for fence in ("````markdown", "~~~python"):
            with self.subTest(fence=fence):
                text = fence + "\n```\n# not a heading\n| a | b |\n| --- | --- |"
                parents, children = split_markdown(text)
                self.assertEqual(len(parents), 1)
                self.assertEqual(children[0].metadata["content_type"], "code")
                self.assertTrue(children[0].metadata["fence_repaired"])
                self.assertNotIn("Header 1", children[0].metadata)
                marker = "````" if fence.startswith("`") else "~~~"
                self.assertEqual(children[0].page_content, text + "\n" + marker)

    def test_oversized_rows_are_not_truncated(self):
        for text, kind in (
            ("```python\n    x = '" + "a" * 100 + "'\n```", "code"),
            ("| a | b |\n| --- | --- |\n| " + "x" * 100 + " | y |", "table"),
            ("| " + "h" * 100 + " |\n| --- |", "table"),
        ):
            with self.subTest(kind=kind):
                parents, children = split_markdown(text, chunk_size=30, chunk_overlap=0, parent_chunk_size=60)
                for documents in (parents, children):
                    self.assertEqual(len(documents), 1)
                    self.assertEqual(documents[0].page_content, text)
                    self.assertTrue(documents[0].metadata["oversized"])
                    self.assertEqual(documents[0].metadata["content_type"], kind)

    def test_escaped_pipes_and_non_table_prose(self):
        table = "| key | value |\n| --- | --- |\n| a\\|b | c |"
        _, children = split_markdown(table + "\n\nnormal | prose\nno separator\n")
        self.assertEqual(children[0].page_content, table)
        self.assertEqual(children[1].metadata["content_type"], "text")

    def test_indented_code_preserves_blank_lines(self):
        code = "    def run():\n        x = 1\n\n        return x\n"
        _, children = split_markdown("说明\n\n" + code + "\n结束",
            chunk_size=30, chunk_overlap=0, parent_chunk_size=60)
        snippets = [c for c in children if c.metadata["content_type"] == "code"]
        self.assertTrue(snippets)
        self.assertEqual("\n".join(c.page_content for c in snippets), code)


class MarkdownImageTests(unittest.TestCase):
    def test_atomic_inline_multiline_and_reference_images(self):
        cases = [
            'before ![图](images/a(b).png "title") after',
            '![图](\n<images/a b.png>\n)',
            '![图][pic]\n\n[pic]: images/a.png "title"',
        ]
        for text in cases:
            with self.subTest(text=text):
                parents, children = split_markdown(text, chunk_size=5, chunk_overlap=0, parent_chunk_size=10)
                for docs in (parents, children):
                    image = next(d for d in docs if d.metadata['content_type'] == 'image')
                    self.assertIn('![图]', image.page_content)
                    self.assertEqual(image.metadata['oversized'], len(image.page_content) > (10 if docs is parents else 5))
                    self.assertEqual(image.page_content, text.split('\n\n')[0])

    def test_code_escapes_and_links_are_not_images(self):
        for text in ['`![图](missing.png)`', r'\![图](missing.png)', '[link](missing.png)', '```md\n![图](missing.png)\n```']:
            with self.subTest(text=text):
                _, children = split_markdown(text)
                self.assertFalse(any(c.metadata['content_type'] == 'image' for c in children))

    def test_encoding_paths_references_batches_and_query(self):
        import torch

        class Model:
            def __init__(self):
                self.calls = []
            def eval(self):
                return self
            def encode(self, **kwargs):
                self.calls.append(kwargs)
                return torch.tensor([[3., 4.]])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['a(b).png', '中文 图.png']:
                (root / name).touch()
            model = Model()
            adapter = VisualizedBGEEmbeddings(model, batch_size=1)
            source = 'before ![图](a(b).png "title") after\n\n![][ref]\n\n[ref]: <中文 图.png>'
            path = root / 'source.md'
            path.write_text(source, encoding='utf-8')
            result = create_block(path, embeddings=adapter)
            self.assertEqual(len(model.calls), len(result.child_documents))
            calls = [c for c in model.calls if 'image' in c]
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]['image'], str((root / 'a(b).png').resolve()))
            self.assertIn('before', calls[0]['text'])
            self.assertNotIn('a(b).png', calls[0]['text'])
            self.assertEqual(calls[1], {'image': str((root / '中文 图.png').resolve())})
            self.assertEqual(adapter.embed_documents([]), [])
            self.assertEqual(adapter.embed_query('query'), [3., 4.])
            vectors = adapter.for_markdown(path, source).embed_documents(['![](a(b).png) ![](<中文 图.png>)'])
            self.assertAlmostEqual(sum(v*v for v in vectors[0]), 1., places=5)
            scoped = adapter.for_markdown(path, source)
            with self.assertRaises(FileNotFoundError):
                scoped.embed_documents(['![](missing.png)'])
            with self.assertRaises(ValueError):
                scoped.embed_documents(['![](https://example.com/a.png)'])


if __name__ == "__main__":
    unittest.main()
