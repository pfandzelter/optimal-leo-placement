# QoS-Aware Resource Placement for the LEO Edge

Placement algorithms, simulation environment, and analysis tools for QoS-aware resource placement on LEO satellite constellations.

If you use this software in a publication, please cite it as:

### Text

T. Pfandzelter and D. Bermbach, **QoS-Aware Resource Placement for LEO Satellite Edge Computing**, 6th IEEE International Conference on Fog and Edge Computing (ICFEC 2022), Taormina, Italy, 2022

### BibTeX

```bibtex
@inproceedings{pfandzelter_optimal:_2022,
    title = "QoS-Aware Resource Placement for LEO Satellite Edge Computing",
    booktitle = "6th IEEE International Conference on Fog and Edge Computing (ICFEC 2022)",
    author = "Pfandzelter, Tobias and Bermbach, David",
    year = 2022,
    publisher = "IEEE",
}
```

A preprint is available on [arXiv](https://arxiv.org/abs/2201.05872).
For a full list of publications, please see [our website](https://www.tu.berlin/en/mcc/research/publications/).

## License

The code in this repository is licensed under the terms of the [MIT](./LICENSE) license.

All code in the `simulation` folder is based on the [SILLEO-SCNS project](https://github.com/Ben-Kempton/SILLEO-SCNS) and licensed under the [GNU General Public License Version 3](./simulation/LICENSE).

## Usage

This project requires Python 3.9 or later.
Install the required packages with `pip install -r requirements.txt`, a virtual environment is recommended.
Alternatively, use the provided Dockerfile and run all subsequent commands inside the container:

```sh
docker build -t py .
docker run --rm -it -v "$(pwd)":/run py
cd /run
```

Please bear in mind that the code in this repository is not optimized for performance.
Memory, disk, and CPU usage can be high, especially for larger constellations and with increased simulation granularity.
The simulations in our paper were performed on an `r6g.16xlarge` EC2 instance with a 200GB SSD in the AWS Frankfurt region.

### Configuration

Configure simulation parameters in `config.py`.
This includes parameters for the shells as well as service level objectives (SLO).

### Placement

The placement algorithm can be found in `placement.py`.
To run the algorthm for the configured shells and SLOs, run `python3 shells.py`.

### Constellation Simulation

The constellation simulation yields ISL distances for the configured shells.
Run `python3 simulation.py` to generate these distances.

### SLO Analysis

Run `python3 placement-distances.py` to analyze the distances between shells and resource nodes over time.

### Summarize

To summarize the results, run `python3 summarize.py`.
Results are then available in `results.csv` (depending on your configuration in `config.py`).

### Visualize

To generate graphs for this information, run `python3 graphs.py`.
