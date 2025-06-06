def main():
    import sys

    if len(sys.argv) == 1:
        print("No arguments provided. Please run `cdhdashboard run`.")
    elif sys.argv[1] == "run":
        run(*sys.argv[2:])
    else:
        print(f"Unknown argument {sys.argv[1]}.")


def run(*args):
    from streamlit.web import cli as stcli
    import sys
    import os

    print("Running app.")
    print(args)
    filename = os.path.join(os.path.dirname(__file__), "../../vd_app.py")
    sys.argv = ["streamlit", "run"]
    if "--" in args:
        try:
            script_args_index = args.index("--")
            script_args = args[script_args_index + 1:]
            sys.argv.extend(args[0:script_args_index])
            sys.argv.append(filename)
            sys.argv.append("""--""")
            sys.argv.extend(script_args)
        except IndexError:
                sys.argv.extend(args[1:])
    else:
        sys.argv.extend(list(args))
        sys.argv.append(filename)

    print(sys.argv)
    sys.exit(stcli.main())

if __name__ == '__main__':
    main()
