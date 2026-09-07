#!/usr/bin/env python
import spine
import yaml
from spine.driver import Driver
from spine.utils.globals import VALUE_COL
from matplotlib import pyplot as plt

# === CHANGE THIS TO YOUR ACTUAL FILE ===
DETECTOR = 'protodune-vd'
DATA_PATH = '/home/nazanin/spine_workshop/protodune-vd_small.root'


cfg = """
base:
  verbosity: info
geo:
  detector: DETECTOR
io:
  loader:
    batch_size: 4
    shuffle: False
    num_workers: 4
    collate_fn: all
    dataset:
      name: larcv
      file_keys: DATA_PATH
      limit_num_files: 10
      #entry_list: [6436, 562, 3802, 6175, 15256] # can also be specified as a file
      #skip_entry_list: [12, 354] # can also be specified as a file
      schema:
        input_data:
          parser: sparse3d
          sparse_event: sparse3d_pcluster
        seg_label:
          parser: sparse3d
          sparse_event: sparse3d_pcluster_semantics
        clust_label:
          parser: cluster3d
          cluster_event: cluster3d_pcluster
          particle_event: particle_corrected
          sparse_semantics_event: sparse3d_pcluster_semantics
          sparse_value_event: sparse3d_pcluster
          add_particle_info: true
          clean_data: true
        ppn_label:
          parser: particle_points
          particle_event: particle_corrected
          sparse_event: sparse3d_pcluster
        meta:
          parser: meta
          sparse_event: sparse3d_pcluster
        run_info:
          parser: meta
          sparse_event: sparse3d_pcluster
""".replace('DETECTOR', DETECTOR).replace('DATA_PATH', DATA_PATH)

cfg = yaml.safe_load(cfg)

# prepare function configures necessary "handlers"
driver = Driver(cfg)
data = driver.process()
print(f"Loaded batch with keys: {data.keys()}")
print(f"clust_label tensor shape: {data['clust_label'].tensor.shape}")

# Create plot and SAVE to file
plt.figure(figsize=(10, 6))
plt.hist(data['clust_label'].tensor[:, VALUE_COL], range=[0,100], bins=50, 
         histtype='step', linewidth=2)
plt.xlabel('Energy [MeV]')
plt.ylabel('Space points')
plt.grid(True)
plt.title('Energy Distribution - ProtoDUNE-VD')

# Save instead of show
plt.savefig('energy_distribution.png', dpi=150, bbox_inches='tight')
print("Plot saved to energy_distribution.png")

# Also print some statistics
import numpy as np
energies = data['clust_label'].tensor[:, VALUE_COL]
print(f"\nEnergy statistics:")
print(f"  Min: {energies.min():.2f} MeV")
print(f"  Max: {energies.max():.2f} MeV")
print(f"  Mean: {energies.mean():.2f} MeV")
print(f"  Median: {np.median(energies):.2f} MeV")
print(f"  Number of voxels: {len(energies)}")

