from patent_gap.cli import main

if __name__ == "__main__":
    import sys

    main(["preprocess", *sys.argv[1:]])

