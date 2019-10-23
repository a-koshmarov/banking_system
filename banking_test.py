from banking import DebitAccount, BankingSystem
from unittest import TestCase


class DebitAccountTest(TestCase):
    def setUp(self):
        self.account = DebitAccount("01.02.2000", "Patrick")

    def test_withdraw(self):
        self.account.balance = 10

        self.account.withdraw(10)
        self.assertEqual(0, self.account.balance)

        self.account.withdraw(10)
        self.assertEqual(0, self.account.balance)

    def test_add_interest(self):
        self.account.balance = 10
        self.account.add_interest(0.1)
        self.assertEqual(11, self.account.balance)


class BankingSystemTest(TestCase):
    def setUp(self):
        self.system = BankingSystem()
          
    def test_add_account(self):
        with self.assertRaises(TypeError):
            self.system.add_account(23)
