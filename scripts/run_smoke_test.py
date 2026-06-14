from patent_gap.cli import main

if __name__ == "__main__":
    main(["compute-gap", "--config", "configs/experiment/synthetic_smoke.yaml"])
    main(["rank-views", "--config", "configs/experiment/synthetic_smoke.yaml"])
    main(["closed-loop", "--config", "configs/experiment/synthetic_smoke.yaml", "simulation.steps=5"])

