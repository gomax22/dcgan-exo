import os
import argparse
from astropy.io import fits
import itertools
import numpy as np
import pickle
from pathlib import Path
from classes.night import Night

from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to data directory")
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-s", "--samples_per_night", required=True, type=int, default=58, help="number of samples to generate per night")
ap.add_argument("-k", "--k", required=True, default=5, type=int, help="number of observations to combine")
ap.add_argument("-b", "--cut-begin", required=False, default=None, help="cut beginning of wavelength range")
ap.add_argument("-e", "--cut-end", required=False, default=None, help="cut end of wavelength range")
args = vars(ap.parse_args())

dataset = args["dataset"]
output = args["output"]
cut_begin = args["cut_begin"]
cut_end = args["cut_end"]
samples_per_night = args["samples_per_night"]
k = args["k"]

# print input parameters
print(f"Dataset: {dataset}")
print(f"Output: {output}")
print(f"Samples per night: {samples_per_night}")
print(f"K: {k}")
print(f"Cut begin: {cut_begin}")
print(f"Cut end: {cut_end}")

# create output directory
if not Path(output).exists():
    Path(output).mkdir(parents=True, exist_ok=True)

# list nights stored in the data directory
night_paths = [os.path.join(dataset, entry) for entry in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, entry)) and entry != "tars"]
print(f"Found {len(night_paths)} nights in the dataset directory")

for night_path in night_paths:
    print(f"\nProcessing night {night_path}")

    # get observations from night
    night = Night.from_directory(night_path, cut_begin, cut_end)
    
    # interpolate the night observations
    night.interpolate()
    
    # cutoff wavelength ranges eventually
    night.cutoff()
    
    # generate samples of night from this night
    date = night_path.split(os.sep)[-1]
    generated_nights = night.generate(k=k, samples_per_night=samples_per_night, out_path=output, date=date)
    
    """
    # save generated nights using np.savedz_compressed
    
    pbar = tqdm(total=len(generated_nights), desc="Saving generated nights...")
    
    for idx, generated_night in enumerate(generated_nights):
        generated_night.save(output, date, idx)
        
        pbar.update(1)
    pbar.close()
    """    
    
    