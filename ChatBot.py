import pandas as pd


def handle_commands(command, args=None):
    if command == "/top_company":
        return get_top_company()
    elif command == "/analyze_role":
        return analyze_role(args)
    elif command == "/company_info":
        return get_top_company(args)
    else:
        return


def get_top_company():
    pass


def analyze_role(args):
    pass


def get_company_info(args):
    pass
