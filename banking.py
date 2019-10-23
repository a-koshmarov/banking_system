class BankingSystem:
    def __init__(self):
        self.accounts = {
            "Account": [],
            "DebitAccount": [],
            "CreditAccount": [],
        }

    def __repr__(self):
        acc_len = len(self.accounts["Account"])
        debit_len = len(self.accounts["DebitAccount"])
        credit_len = len(self.accounts["CreditAccount"])
        s = '''Banking system has {} regular accounts, {} debit accounts
        and {} credit accounts'''.format(acc_len, debit_len, credit_len)
        return s

    def add_account(self, account):
        if type(account) is Account:
            self.accounts["Account"].append(account)
        elif type(account) is DebitAccount:
            self.accounts["DebitAccount"].append(account)
        elif type(account) is CreditAccount:
            self.accounts["CreditAccount"].append(account)
        else:
            raise TypeError


class Account:
    def __init__(self, reg_date, name):
        self.reg_date = reg_date
        self.name = name

    def __repr__(self):
        return "{} was registered on {}".format(self.name, self.reg_date)


class DebitAccount(Account):
    def __init__(self, reg_date, name):
        super(DebitAccount, self).__init__(reg_date, name)
        self.balance = 0

    def __repr__(self):
        return "{} was registered on {}. Balance is {}".format(self.name, self.reg_date, self.balance)

    def withdraw(self, amount):
        """A function to withdraw money from debit account
        
        Arguments:
            amount {float} -- An amount of money to be withdrawn
        """
        
        if (self.balance - amount) >= 0:
            self.balance -= amount
        else:
            print("Insufficient funds")

    def deposit(self, amount):
        self.balance += amount

    def add_interest(self, percent):
        self.balance *= 1 + percent

    def transfer(self, account, amount, fee):
        if fee:
            self.balance -= amount * (1+fee)

        else:
            self.balance -= amount
        account.balance += amount

class CreditAccount(Account):
    def __init__(self, reg_date, name):
        super(CreditAccount, self).__init__(reg_date, name)
        self.debt = 0

    def add_interest(self, percent):
        self.debt *= 1 + percent

def main():
    bank = BankingSystem()
    account1 = DebitAccount("02.04.2010", "Bob")
    account2 = DebitAccount("02.04.2012", "Jo")

    bank.add_account(account1)
    bank.add_account(account2)
    print(bank)


if __name__ == "__main__":
    main()
