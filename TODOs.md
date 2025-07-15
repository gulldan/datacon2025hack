# TODO List for Drug Discovery Pipeline

## V. DOCKING (STRUCTURAL VALIDATION)

### 🔹 8. Preparation for Docking

#### Select PDB Structure
Choose from the following options:
- **4YU2**: Basic structure
- **3ANQ**: Structure in complex with inhibitor
- **3ANR**: Structure in complex with harmine
- **5AIK**: Structure in complex with inhibitor LDN-211898
- **4AZE**: Structure in complex with inhibitor Leucettine L41
- **2WO6**: Another frequently mentioned structure
- **6A1G**: Structure in complex with compound 32

#### Protein Processing
- Remove water molecules
- Remove protons
- Save as PDBQT format

#### Ligand Preparation
- Convert molecules to 3D (RDKit + ETKDG)
- Save as PDBQT format

### 🔹 9. AutoDock Vina Execution

#### Pocket Definition
- Set center and size of binding pocket (from PDB structure)

#### Docking Process
- Run docking (batch processing)
- Collect binding scores and poses

## VI. HIT SELECTION AND DOCUMENTATION

### 🔹 10. Final Selection

Select molecules with the following criteria:

- **Docking score** < –6.5 kcal/mol
- **QED** > 0.6
- **ML prediction** → activity
- **BBB permeability** (TPSA < 90, logP normal)
- **SA** < 5

#### Visualization
- Generate 2D and 3D visualizations

### 🔹 11. Results Documentation

#### Output Tables
- Table with: SMILES, QED, SA, TPSA, IC₅₀, Docking Score

#### Scripts
- Generation + filtering script

#### Documentation
- README with description of each stage
- Example of 5–10 hit molecules (with visualization)
