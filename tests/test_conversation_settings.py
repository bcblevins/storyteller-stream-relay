import unittest
from fastapi import HTTPException
from app import resolve_temperature, validate_temperature


class ConversationSettingsTests(unittest.TestCase):
    def test_temperature_override_preserves_zero_and_bot_default(self):
        self.assertEqual(resolve_temperature({"temperature": 0}, {"temperature": 0.8}), 0)
        self.assertEqual(resolve_temperature({}, {"temperature": 0.8}), 0.8)
        self.assertEqual(resolve_temperature({}, {}), 0.1)

    def test_invalid_temperature_is_rejected(self):
        for value in [None, True, "0.5", -0.1, 2.1, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(HTTPException):
                validate_temperature({"temperature": value})
        for value in [0, 0.7, 2]:
            validate_temperature({"temperature": value})
        validate_temperature({})
