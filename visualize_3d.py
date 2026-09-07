#!/usr/bin/env python
import spine
import yaml
from spine.driver import Driver
from spine.utils.globals import VALUE_COL, SHAPE_COL, PPN_LTYPE_COL, GROUP_COL, INTER_COL, NU_COL, GHOST_SHP
from plotly import graph_objs as go
from plotly.offline import plot  # Changed from iplot to plot for file output

from spine.vis.geo import GeoDrawer
from spine.vis.point import scatter_points
from spine.vis.layout import layout3d, PLOTLY_COLORS, HIGH_CONTRAST_COLORS

# === CONFIGURATION ===
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
    num_workers: 2
    collate_fn: all
    dataset:
      name: larcv
      file_keys: DATA_PATH
      limit_num_files: 1
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
""".replace('DETECTOR', DETECTOR).replace('DATA_PATH', DATA_PATH)

print("Loading data...")
cfg = yaml.safe_load(cfg)
driver = Driver(cfg)
data = driver.process()

# Select first entry (entry 0)
entry = 0
print(f"\nProcessing entry {entry}...")

# Extract data for this entry
clust_label = data['clust_label'][entry]
input_data = data['input_data'][entry]
seg_label = data['seg_label'][entry][:, SHAPE_COL]
ppn_label = data['ppn_label'][entry]

# Get detector geometry drawer
geo_drawer = GeoDrawer(detector_coords=False)
tpc_traces = geo_drawer.tpc_traces(meta=data['meta'][entry])
opt_traces = geo_drawer.optical_traces(meta=data['meta'][entry])
crt_traces = geo_drawer.crt_traces(meta=data['meta'][entry])

nonghost_mask = seg_label < GHOST_SHP

print("\n=== Generating 3D Visualizations ===")
print("This will create HTML files you can open in your browser")

# ============================================
# 1. INPUT DATA VISUALIZATION
# ============================================
print("\n1. Creating input charge visualization...")
trace1 = []

trace1 += scatter_points(input_data, color=input_data[:, VALUE_COL],
                        markersize=2, cmin=0, cmax=50, colorscale='Inferno', 
                        name='Input charge')

trace1 += scatter_points(input_data[nonghost_mask],
                        color=input_data[nonghost_mask, VALUE_COL],
                        markersize=2, cmin=0, cmax=50, colorscale='Inferno', 
                        name='Input charge (true non-ghost)')

trace1 += tpc_traces
trace1 += opt_traces
trace1 += crt_traces

fig1 = go.Figure(data=trace1, layout=layout3d(use_geo=True, detector_coords=False, 
                                              meta=data['meta'][entry], show_crt=True))
plot(fig1, filename='1_input_charge.html', auto_open=False)
print("   Saved to: 1_input_charge.html")

# ============================================
# 2. SEMANTIC LABELS
# ============================================
print("\n2. Creating semantic labels visualization...")
trace2 = []

trace2 += scatter_points(input_data, color=seg_label,
                        markersize=2, cmin=0, cmax=5, colorscale=PLOTLY_COLORS[:6],
                        name='Seg. labels (no ghosts)')

trace2 += tpc_traces

fig2 = go.Figure(data=trace2, layout=layout3d(meta=data['meta'][entry]))
plot(fig2, filename='2_semantic_labels.html', auto_open=False)
print("   Saved to: 2_semantic_labels.html")

# ============================================
# 3. POINTS OF INTEREST (PPN)
# ============================================
print("\n3. Creating points of interest visualization...")
trace3 = []

trace3 += scatter_points(input_data, color=seg_label,
                        markersize=1, cmin=0, cmax=5, colorscale=PLOTLY_COLORS[:6],
                        name='Seg. labels (no ghosts)')

trace3 += scatter_points(ppn_label, color=ppn_label[:, PPN_LTYPE_COL],
                        markersize=5, cmin=0, cmax=5, colorscale=PLOTLY_COLORS[:6])
trace3[-1].name = "True point labels"

trace3 += tpc_traces

fig3 = go.Figure(data=trace3, layout=layout3d(meta=data['meta'][entry]))
plot(fig3, filename='3_points_of_interest.html', auto_open=False)
print("   Saved to: 3_points_of_interest.html")

# ============================================
# 4. PARTICLE INSTANCES
# ============================================
print("\n4. Creating particle instances visualization...")
trace4 = []

trace4 += scatter_points(clust_label, color=clust_label[:, GROUP_COL],
                        markersize=2, cmin=0, cmax=50, colorscale=HIGH_CONTRAST_COLORS,
                        name='True group labels')

trace4 += tpc_traces

fig4 = go.Figure(data=trace4, layout=layout3d(meta=data['meta'][entry]))
plot(fig4, filename='4_particle_instances.html', auto_open=False)
print("   Saved to: 4_particle_instances.html")

# ============================================
# 5. INTERACTION GROUPS
# ============================================
print("\n5. Creating interaction groups visualization...")
trace5 = []

trace5 += scatter_points(clust_label, color=clust_label[:, INTER_COL],
                        markersize=2, cmin=0, cmax=50, colorscale=HIGH_CONTRAST_COLORS)
trace5[-1].name = 'True interaction labels'

trace5 += tpc_traces

fig5 = go.Figure(data=trace5, layout=layout3d(meta=data['meta'][entry]))
plot(fig5, filename='5_interaction_groups.html', auto_open=False)
print("   Saved to: 5_interaction_groups.html")

# ============================================
# 6. NEUTRINO VS COSMICS
# ============================================
print("\n6. Creating neutrino vs cosmics visualization...")
trace6 = []

trace6 += scatter_points(clust_label, color=clust_label[:, NU_COL],
                        markersize=2, cmin=-1, cmax=0, colorscale='Portland',
                        name='Nu / cosmic labels')

trace6 += tpc_traces

fig6 = go.Figure(data=trace6, layout=layout3d(meta=data['meta'][entry]))
plot(fig6, filename='6_neutrino_vs_cosmics.html', auto_open=False)
print("   Saved to: 6_neutrino_vs_cosmics.html")

print("\n=== All visualizations complete! ===")
print(f"Files saved in: {__file__}")
print("\nTo view: Open the .html files in your browser")
print("Or run: firefox *.html  (or your browser of choice)")
