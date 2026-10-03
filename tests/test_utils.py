import unittest

from src.utils import is_valid_model_payload


class ModelPayloadValidationTests(unittest.TestCase):
    def test_rejects_html_error_response(self) -> None:
        payload = b'<!DOCTYPE html><html><body>404 Not Found</body></html>'
        self.assertFalse(is_valid_model_payload(payload))

    def test_accepts_binary_onnx_like_payload(self) -> None:
        payload = b'\x08\x01\x12\x00' + b'x' * 200000
        self.assertTrue(is_valid_model_payload(payload))


if __name__ == "__main__":
    unittest.main()
