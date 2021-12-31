# Anxiety-Detection
Code for Anxiety Level Detection from physiological signals using supervised ML algorithms

### How to run the program


1. Clone the repo
   ```sh
   git clone https://github.com/sidesh27/Anxiety-Detection.git
   ```
   For accounts that are SSH configured
   ```sh
    git clone git@github.com:sidesh27/Anxiety-Detection.git
   ```
2. Install pip
   ```sh
   python -m pip install --upgrade pip
   ```
3. Create and Activate Virtual Environment (Linux)
   ```sh
   python3 -m venv [environment-name]
   source [environment-name]/bin/activate
   ```
4. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```
5. Run main
   ```sh
   python3 main.py --option value
   ```
   
6. The following are the list of trainable parameters that can be provided in the terminal

| Option               | Description                                                                    |
| :------------------- | :----------------------------------------------------------------------------- |
| `--detector or -d`     | R-peak Detection Algorithms [pan-tompkins, hamilton] -> string |
| `--classifier or -clf`  | Classification Algorithms [logreg, decisiontree, xgboost, randomforest, extratrees, bagging] -> string                           |


## References

[1] Michał Sznajder, & Marta Łukowska. (2017). Python Online and Offline ECG QRS Detector based on the Pan-Tomkins algorithm (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.826614 