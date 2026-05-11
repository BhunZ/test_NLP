import unittest

from backend import rag_service


class TestRagUtils(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(rag_service._format_timestamp(0), "00:00")
        self.assertEqual(rag_service._format_timestamp(5.9), "00:05")
        self.assertEqual(rag_service._format_timestamp(65.2), "01:05")

    def test_youtube_helpers(self):
        self.assertEqual(
            rag_service._youtube_thumbnail("abc"),
            "https://img.youtube.com/vi/abc/mqdefault.jpg",
        )
        self.assertEqual(
            rag_service._youtube_url("abc", 42),
            "https://youtu.be/abc?t=42",
        )

    def test_parse_citations(self):
        text = "A [1] B [2] C [1]"
        self.assertEqual(rag_service._parse_citation_numbers(text), [1, 2])


if __name__ == "__main__":
    unittest.main()

