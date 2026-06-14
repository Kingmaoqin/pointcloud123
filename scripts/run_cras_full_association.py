from patent_gap.cli import main

if __name__ == "__main__":
    import sys

    main(["associate-cras", "--config", "configs/experiment/cras_full.yaml", *sys.argv[1:]])

