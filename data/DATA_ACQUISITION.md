# Data Acquisition Guide for pmp_research

This guide shows you exactly how to download/obtain all files needed for the `data/raw/` folder.

---

## 📋 Overview

You need to populate 4 subdirectories:
```
data/raw/
├── pdb/              ← PDB structure files
├── opm/              ← OPM membrane orientation data
├── md_trajectories/  ← MDAnalysis-compatible trajectory files
└── labels/           ← Binding residue annotations (CSV)
```

---

## 1️⃣ PDB Files (`data/raw/pdb/`)

### Option A: Download from RCSB PDB (Recommended for small datasets)

**For individual proteins:**

```bash
# Download a single structure
wget https://files.rcsb.org/download/2ABC.pdb -O data/raw/pdb/2abc.pdb

# Or use curl
curl https://files.rcsb.org/download/2ABC.pdb -o data/raw/pdb/2abc.pdb

# Download as mmCIF format (more modern)
wget https://files.rcsb.org/download/2ABC.cif -O data/raw/pdb/2abc.cif
```

**For a list of proteins:**

```bash
# Create file: pdb_list.txt
cat > pdb_list.txt << 'EOF'
2ABC
1OPZ
1M6G
4IU1
2KFX
EOF

# Download all
while read pdb_id; do
  wget https://files.rcsb.org/download/${pdb_id}.pdb \
    -O data/raw/pdb/${pdb_id,,}.pdb
done < pdb_list.txt
```

### Option B: Batch download from PDBTools

```bash
pip install pdbtools

# Download multiple structures
python << 'PYEOF'
from pdbtools import pdb_download

pdb_ids = ['2ABC', '1OPZ', '1M6G', '4IU1', '2KFX']
for pdb_id in pdb_ids:
    pdb_download.download(pdb_id.upper(), target_dir='data/raw/pdb')
PYEOF
```

### Option C: Use BioPython

```bash
pip install biopython

python << 'PYEOF'
from Bio.PDB import PDBList

pdbl = PDBList()
pdb_ids = ['2ABC', '1OPZ', '1M6G', '4IU1', '2KFX']

for pdb_id in pdb_ids:
    structure = pdbl.retrieve_pdb_file(
        pdb_id,
        pdir='data/raw/pdb',
        file_format='pdb'
    )
    # Rename from uppercase default
    import shutil, os
    new_name = f"data/raw/pdb/{pdb_id.lower()}.pdb"
    if os.path.exists(new_name):
        os.remove(new_name)
    shutil.move(structure, new_name)
PYEOF
```

**File format:** `.pdb` (text) or `.cif` (text, more modern)

**Location:** `data/raw/pdb/2abc.pdb` (lowercase codes)

---

## 2️⃣ OPM Data (`data/raw/opm/`)

OPM (Orientations of Proteins in Membranes) provides **membrane normal vectors** and **depth annotations**.

### Option A: Download from OPM Web Server (Manual)

1. Go to https://opm.phar.umich.edu/
2. Search for your protein (e.g., "2ABC")
3. Click "Download" → "Annotation for this entry"
4. Save as `data/raw/opm/2abc.json`

### Option B: Use OPM API (Recommended)

```bash
pip install requests

python << 'PYEOF'
import requests
import json
import os

pdb_ids = ['2ABC', '1OPZ', '1M6G', '4IU1', '2KFX']

for pdb_id in pdb_ids.lower():
    # OPM API endpoint
    url = f"https://opm.phar.umich.edu/api/proteins/{pdb_id}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Extract key fields
            opm_entry = {
                'pdb_id': pdb_id,
                'normal': data.get('membrane_normal', [0, 0, 1]),  # (x, y, z)
                'thickness': data.get('thickness', 30.0),          # Angstroms
                'tilt_angle': data.get('tilt_angle', 0.0),
                'depth_profile': data.get('depth_profile', {}),
            }
            
            # Save to JSON
            output_path = f'data/raw/opm/{pdb_id.lower()}.json'
            with open(output_path, 'w') as f:
                json.dump(opm_entry, f, indent=2)
            
            print(f"✓ {pdb_id}: normal={opm_entry['normal']}")
        else:
            print(f"✗ {pdb_id}: HTTP {response.status_code}")
    except Exception as e:
        print(f"✗ {pdb_id}: {e}")
PYEOF
```

### Option C: Compute Membrane Normal from PDB (Fallback)

If OPM data unavailable, estimate from structure:

```bash
pip install tmtools

python << 'PYEOF'
import numpy as np
from Bio.PDB import PDBParser

def estimate_membrane_normal(pdb_file):
    """
    Heuristic: membrane normal ≈ principal axis of residue spread.
    For transmembrane proteins, Z-axis is typically the membrane normal.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_file)
    
    # Get all Cα coordinates
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
    
    ca_coords = np.array(ca_coords)
    
    # Compute covariance matrix
    cov = np.cov(ca_coords.T)
    
    # Eigenvectors (principal axes)
    evals, evecs = np.linalg.eig(cov)
    
    # Smallest eigenvalue → membrane normal (thinest axis)
    min_idx = np.argmin(evals)
    normal = evecs[:, min_idx]
    
    # Ensure Z-component is positive
    if normal[2] < 0:
        normal = -normal
    
    return normal / np.linalg.norm(normal)

# Process all proteins
import json
import glob

for pdb_file in glob.glob('data/raw/pdb/*.pdb'):
    pdb_id = pdb_file.split('/')[-1].replace('.pdb', '')
    
    normal = estimate_membrane_normal(pdb_file)
    
    opm_data = {
        'pdb_id': pdb_id,
        'normal': normal.tolist(),
        'thickness': 30.0,
        'tilt_angle': 0.0,
        'source': 'estimated_from_pdb',
    }
    
    output_path = f'data/raw/opm/{pdb_id}.json'
    with open(output_path, 'w') as f:
        json.dump(opm_data, f, indent=2)
    
    print(f"✓ {pdb_id}: normal={normal}")
PYEOF
```

**File format:** `.json` with keys:
```json
{
  "pdb_id": "2abc",
  "normal": [0.0, 0.0, 1.0],
  "thickness": 30.0,
  "tilt_angle": 0.0
}
```

**Location:** `data/raw/opm/2abc.json`

---

## 3️⃣ MD Trajectories (`data/raw/md_trajectories/`)

MD simulation trajectories are **optional but highly recommended** for Phase 1 training.

### Option A: Use Published MD Data (MDTRAJ Database)

Many proteins have published MD trajectories:

```bash
pip install mdanalysis

python << 'PYEOF'
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD

# Example: using MDAnalysis test data
u = mda.Universe(PSF, DCD)

# Save selected frames as XTC (compact format)
protein = u.select_atoms('protein')
protein.write('data/raw/md_trajectories/2abc.xtc')
PYEOF
```

### Option B: Run Your Own MD Simulation (Advanced)

```bash
pip install gromacs openmm

# Use OpenMM to run ~100ns MD
python << 'PYEOF'
from openmm import *
from openmm.app import *
from openmm.unit import *
import mdtraj

# Load PDB
pdb = PDBFile('data/raw/pdb/2abc.pdb')

# Create system
ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
modeller = Modeller(pdb.topology, pdb.positions)

# Add water box
modeller.addSolvent(ff, boxSize=Vector(5, 5, 5)*nanometers)

# Create simulation
system = ff.createSystem(modeller.topology)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

# Minimise energy
simulation.minimizeEnergy()

# Run MD (save every 10 frames)
simulation.reporters.append(
    mdtraj.reporters.MDTrajReporter(
        'data/raw/md_trajectories/2abc.xtc',
        10
    )
)
simulation.reporters.append(StateDataReporter('output.log', 1000))

simulation.step(100000)  # 200 ps
PYEOF
```

### Option C: Use Pre-computed MD Data (RCSB PDB-Dev)

```bash
# Some proteins have pre-computed MD data from RCSB
wget https://pdb-dev.rcsb.org/pdb/2abc/md_trajectory.xtc \
  -O data/raw/md_trajectories/2abc.xtc
```

**File format:** `.xtc` (XTC is most common, also `.dcd`, `.trr`)

**How to check MD file:**

```python
import MDAnalysis as mda

u = mda.Universe('data/raw/pdb/2abc.pdb', 'data/raw/md_trajectories/2abc.xtc')
print(f"Frames: {u.trajectory.n_frames}")
print(f"Atoms: {u.trajectory.n_atoms}")
print(f"Time span: {u.trajectory.time} ps")
```

**Location:** `data/raw/md_trajectories/2abc.xtc`

---

## 4️⃣ Binding Labels (`data/raw/labels/`)

Annotate which residues bind the membrane. This is **required for training**.

### Option A: Manual Annotation from Literature

Read the paper and identify binding residues:

```bash
mkdir -p data/raw/labels

# Create CSV: residue_id, is_binding
cat > data/raw/labels/2abc_binding.csv << 'EOF'
residue_id,is_binding
1,0
2,0
3,1
4,1
5,0
...
EOF
```

### Option B: Automated from MD Simulation

Identify residues that contact the membrane in MD:

```bash
pip install mdanalysis

python << 'PYEOF'
import MDAnalysis as mda
import numpy as np
import pandas as pd

u = mda.Universe('data/raw/pdb/2abc.pdb', 'data/raw/md_trajectories/2abc.xtc')
protein = u.select_atoms('protein')

# Membrane boundaries (from OPM data)
membrane_z_min = -15.0  # Angstroms
membrane_z_max = 15.0

# Count frames where each residue contacts membrane
contact_count = {res.resid: 0 for res in protein.residues}
total_frames = 0

for frame in u.trajectory:
    total_frames += 1
    for residue in protein.residues:
        # Get residue center of mass
        com = residue.atoms.center_of_mass()
        
        # Check if in membrane zone
        if membrane_z_min <= com[2] <= membrane_z_max:
            contact_count[residue.resid] += 1

# Threshold: residues in membrane > 50% of frames
threshold = total_frames * 0.5

labels = []
for residue in protein.residues:
    is_binding = 1 if contact_count[residue.resid] > threshold else 0
    labels.append({
        'residue_id': residue.resid,
        'is_binding': is_binding,
    })

# Save CSV
df = pd.DataFrame(labels)
df.to_csv('data/raw/labels/2abc_binding.csv', index=False)

print(f"Binding residues: {df['is_binding'].sum()} / {len(df)}")
PYEOF
```

### Option C: From PDBePISA (Interface Prediction)

PDBePISA predicts protein-membrane interfaces:

```bash
# Download prediction from PDBePISA
# Go to: https://www.ebi.ac.uk/pdbe/pisa/
# Search protein → download interface residues

# Or use command-line tool:
wget https://www.ebi.ac.uk/pdbe/pisa/pisa_cgi.cgi?2abc \
  -O /tmp/pisa_2abc.html

# Parse HTML and extract interface residues (manual parsing needed)
```

**File format:** CSV with columns:
```csv
residue_id,is_binding
1,0
2,0
3,1
4,1
5,0
```

**Location:** `data/raw/labels/2abc_binding.csv`

---

## 🔄 Complete Workflow: Download Everything

Here's a **complete bash script** to download all data:

```bash
#!/bin/bash
# download_data.sh

set -e  # Exit on error

mkdir -p data/raw/{pdb,opm,md_trajectories,labels}
mkdir -p data/splits

# List of PMP proteins to download
PDB_IDS=(
  "2ABC"  # AAGAB
  "1OPZ"  # OmpF porin
  "1M6G"  # OmpG porin
  "4IU1"  # OmpW
  "2KFX"  # Lipocalin
  "1QQU"  # OmpA
  "1BXW"  # OmpX
  "2K0M"  # FhaC
  "1BY5"  # Laminin γ1
  "3B5D"  # Leukotriene C4 synthase
)

echo "📥 Downloading PDB files..."
for pdb_id in "${PDB_IDS[@]}"; do
  pdb_lower=$(echo $pdb_id | tr '[:upper:]' '[:lower:]')
  echo "  ⏳ $pdb_id..."
  wget -q https://files.rcsb.org/download/${pdb_id}.pdb \
    -O data/raw/pdb/${pdb_lower}.pdb || echo "    ✗ Failed"
done
echo "✓ PDB download complete"

echo ""
echo "🧭 Downloading OPM data..."
python << 'PYEOF'
import requests
import json

PDB_IDS = ["2ABC", "1OPZ", "1M6G", "4IU1", "2KFX", "1QQU", "1BXW", "2K0M", "1BY5", "3B5D"]

for pdb_id in PDB_IDS:
    pdb_lower = pdb_id.lower()
    url = f"https://opm.phar.umich.edu/api/proteins/{pdb_lower}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            opm_entry = {
                'pdb_id': pdb_lower,
                'normal': data.get('membrane_normal', [0, 0, 1]),
                'thickness': data.get('thickness', 30.0),
            }
            with open(f'data/raw/opm/{pdb_lower}.json', 'w') as f:
                json.dump(opm_entry, f, indent=2)
            print(f"  ✓ {pdb_id}")
        else:
            print(f"  ✗ {pdb_id}: HTTP {response.status_code}")
    except Exception as e:
        print(f"  ✗ {pdb_id}: {e}")
PYEOF
echo "✓ OPM download complete"

echo ""
echo "📝 Creating placeholder labels (you need to fill these manually)..."
for pdb_id in "${PDB_IDS[@]}"; do
  pdb_lower=$(echo $pdb_id | tr '[:upper:]' '[:lower:]')
  
  # Create CSV with all residues as non-binding (placeholder)
  cat > data/raw/labels/${pdb_lower}_binding.csv << 'EOF'
residue_id,is_binding
1,0
EOF
  
  echo "  ✓ data/raw/labels/${pdb_lower}_binding.csv (PLACEHOLDER)"
done
echo "✓ Label files created (UPDATE WITH ACTUAL DATA)"

echo ""
echo "📂 Creating train/val/test splits..."
cat > data/splits/train.txt << 'EOF'
2abc
1opz
1m6g
4iu1
2kfx
EOF

cat > data/splits/val.txt << 'EOF'
1qqu
1bxw
EOF

cat > data/splits/test.txt << 'EOF'
2k0m
1by5
3b5d
EOF
echo "✓ Splits created"

echo ""
echo "✅ Data download COMPLETE!"
echo ""
echo "⚠️  NEXT STEPS:"
echo "1. Fill in actual binding labels in data/raw/labels/*.csv"
echo "2. (Optional) Download MD trajectories for each protein"
echo "3. Run: python scripts/preprocess.py"

ls -lh data/raw/*/
```

**Run it:**

```bash
chmod +x download_data.sh
./download_data.sh
```

---

## ✅ Verification Checklist

After downloading, verify all files:

```bash
python << 'PYEOF'
import os
import json
from Bio.PDB import PDBParser

# Expected structure
expected = {
    'pdb': ['2abc.pdb', '1opz.pdb', '1m6g.pdb'],
    'opm': ['2abc.json', '1opz.json', '1m6g.json'],
    'labels': ['2abc_binding.csv', '1opz_binding.csv'],
}

all_ok = True

for folder, files in expected.items():
    print(f"\n📂 {folder}/")
    for filename in files:
        path = f"data/raw/{folder}/{filename}"
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {filename} ({size:,} bytes)")
            
            # Quick validation
            if folder == 'pdb' and filename.endswith('.pdb'):
                try:
                    parser = PDBParser(QUIET=True)
                    parser.get_structure(filename[:-4], path)
                    print(f"    └─ ✓ Valid PDB")
                except:
                    print(f"    └─ ✗ Invalid PDB")
                    all_ok = False
            
            elif folder == 'opm':
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if 'normal' in data and len(data['normal']) == 3:
                        print(f"    └─ ✓ Valid OPM (normal={data['normal']})")
                    else:
                        print(f"    └─ ✗ Invalid OPM")
                        all_ok = False
                except:
                    print(f"    └─ ✗ Invalid JSON")
                    all_ok = False
        else:
            print(f"  ✗ {filename} NOT FOUND")
            all_ok = False

if all_ok:
    print(f"\n✅ All files present and valid!")
else:
    print(f"\n⚠️  Some files missing or invalid")
PYEOF
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| PDB file won't download | Try mmCIF format instead: `.cif` instead of `.pdb` |
| OPM API returns 404 | Protein may not be in OPM database; estimate normal from PDB |
| MD trajectory file huge | Downsample: `trjconv -f traj.xtc -o traj_sampled.xtc -skip 10` |
| Don't have MD data | Optional! Phase 1 still works with just PDB + OPM. Set `n_snapshots=1` in config |
| Labels unclear | Read the original publication or use automated method (contact with membrane) |

---

## 📚 Public PMP Datasets You Can Use

Several pre-made datasets available:

1. **OPM Database** — https://opm.phar.umich.edu/
   - 1,000+ membrane proteins with annotations
   - Download entire dataset or specific proteins

2. **MemProtMD** — https://memprotmd.bioch.ox.ac.uk/
   - MD-simulated membrane proteins
   - Download trajectories + structures

3. **PDBTM** — http://pdbtm.enzim.hu/
   - 600+ transmembrane protein structures
   - Topology annotations

4. **MPDB** — https://mpdb.toledolab.org/
   - 600+ membrane proteins
   - Ready-to-use labels

---

## 🎯 Final Structure After Download

```
data/raw/
├── pdb/
│   ├── 2abc.pdb              ← 50-100 KB
│   ├── 1opz.pdb
│   └── ...
├── opm/
│   ├── 2abc.json             ← 1-2 KB
│   ├── 1opz.json
│   └── ...
├── md_trajectories/
│   ├── 2abc.xtc              ← 500 MB - 5 GB (optional)
│   ├── 1opz.xtc
│   └── ...
└── labels/
    ├── 2abc_binding.csv      ← 10 KB
    ├── 1opz_binding.csv
    └── ...

Total size: ~50 GB (with MD), ~1 GB (without MD)
```

**Enjoy! 🚀**
