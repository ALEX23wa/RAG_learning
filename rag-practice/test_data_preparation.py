"""运行：python -m unittest discover -s rag-practice -p test_data_preparation.py"""
import tempfile
from pathlib import Path
import unittest

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from data_preparation import (
    Heading, build_chunks, build_parent_context, detect_numbered_headings,
    extract_pdf, join_pages, prepare, resolve_parents,
)


class PreparationTests(unittest.TestCase):
    def fixture(self):
        pages = [{"doc_id": "demo", "source": "demo.pdf", "file_sha256": "hash",
                  "page": i + 1, "text": text} for i, text in enumerate([
                      "Introduction\n1.1 First\n" + "Alpha text. " * 15,
                      "1.1.1 Detail\n" + "Details. " * 15 + "\n1.2 Second\n" + "Beta. " * 15,
                  ])]
        text = join_pages(pages)
        headings = [Heading(text.index(title), title, level, "test") for title, level in
                    [("1.1 First", 2), ("1.1.1 Detail", 3), ("1.2 Second", 2)]]
        return pages, text, headings

    def test_parent_children_and_full_context(self):
        pages, text, headings = self.fixture()
        parents, children = build_chunks(pages, text, headings, child_size=50, overlap=10)
        self.assertEqual(len(parents), 3)
        self.assertEqual("".join(p["page_content"] for p in parents), text)
        self.assertIn("1.1.1 Detail", parents[1]["page_content"])
        self.assertNotIn("1.2 Second", parents[1]["page_content"])
        self.assertEqual(parents[1]["metadata"]["page_end"], 2)
        for child in children:
            m = child["metadata"]
            self.assertEqual(child["page_content"], text[m["start"]:m["end"]])
            self.assertLessEqual(len(child["page_content"]), 50)
        hits = [c for c in children if c["metadata"]["parent_id"] == parents[1]["id"]]
        self.assertEqual(resolve_parents(hits, parents), [parents[1]])
        self.assertIn(parents[1]["page_content"], build_parent_context(hits, parents))
        with self.assertRaises(ValueError):
            build_parent_context(hits, parents, max_chars=10)
        with self.assertRaises(ValueError):
            resolve_parents(hits, [])
        self.assertEqual(build_chunks(pages, text, headings, child_size=50, overlap=10), (parents, children))

    def test_recursive_and_fallback(self):
        pages, text, headings = self.fixture()
        parents, children = build_chunks(pages, text, headings, strategy="recursive", chunk_size=60, overlap=10)
        self.assertEqual(parents, [])
        self.assertTrue(all("parent_id" not in c["metadata"] for c in children))
        parents, _ = build_chunks(pages, text, [])
        self.assertEqual(parents[0]["page_content"], text)
        self.assertEqual(parents[0]["metadata"]["heading_method"], "whole_document_fallback")

    def test_heading_patterns_and_validation(self):
        h = detect_numbered_headings("1.2 Heading\nBody.\n1.2.1 Detail\n第二章 方法\n2.1 Contents .... 9")
        self.assertEqual([x.level for x in h], [2, 3, 1])
        pages, text, headings = self.fixture()
        with self.assertRaises(ValueError):
            build_chunks(pages, text, headings, child_size=50, overlap=50)

    def test_pdf_bookmarks_on_same_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.PDF"
            writer = PdfWriter()
            page = writer.add_blank_page(width=612, height=792)
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                     NameObject('/Subtype'): NameObject('/Type1'),
                                     NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):
                DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(b'BT /F1 12 Tf 50 700 Td (1.1 First) Tj 0 -20 Td (Alpha body) Tj '
                            b'0 -20 Td (1.2 Second) Tj 0 -20 Td (Beta body) Tj ET')
            page[NameObject('/Contents')] = writer._add_object(stream)
            writer.add_outline_item('1.1 First', 0)
            writer.add_outline_item('1.2 Second', 0)
            writer.write(path)
            pages, text, headings, warnings = extract_pdf(path, root)
            self.assertEqual(len(headings), 2)
            self.assertLess(headings[0].offset, headings[1].offset)
            self.assertEqual(warnings, [])
            parents, _ = build_chunks(pages, text, headings)
            self.assertEqual(len(parents), 2)
            self.assertNotIn('Beta', parents[0]['page_content'])
            output = prepare(root, root / 'output', strategy='structure')
            self.assertTrue((output / 'manifest.json').is_file())

    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prepare(Path(tmp), Path(tmp) / 'output')


if __name__ == '__main__':
    unittest.main()
