# SCULPT Data Format Specification

This document describes how to prepare your COLTRIMS (Cold Target Recoil Ion Momentum Spectroscopy) data files for use with SCULPT. The platform supports flexible particle configurations — from simple two-body breakups to complex many-body fragmentation channels.

---

## Supported File Formats

SCULPT accepts the following file types:

| Format | Extension | Separator | Notes |
|--------|-----------|-----------|-------|
| Space-delimited text | `.dat`, `.txt` | Whitespace (spaces/tabs) | Most common for COLTRIMS data |
| Comma-separated values | `.csv` | Comma | Standard CSV format |

Files **must include a header row** with column names as the first line. SCULPT uses the header to identify particle types and momentum components.

---

## Column Ordering

**Columns can appear in any order in your file.** SCULPT identifies particles by matching header names against its naming patterns, not by column position. For example, these two files are treated identically:

```
Px_ion1  Py_ion1  Pz_ion1  Px_electron1  Py_electron1  Pz_electron1
...
```

```
Px_electron1  Pz_ion1  Py_electron1  Px_ion1  Pz_electron1  Py_ion1
...
```

Internally, SCULPT always reorders particles as: **ions → neutrals → electrons → other**, sorted by their index number within each type. Your file just needs correct header names.

---

## Column Naming Convention

Each particle in your data requires three momentum columns: one for each spatial component (x, y, z). SCULPT identifies particles by matching column names against specific patterns.

### Naming Format

```
P{component}_{type}{number}
```

Where:
- **`{component}`** is `x`, `y`, or `z` (case-insensitive)
- **`{type}`** is `ion`, `neutral`, `electron`, or `particle`
- **`{number}`** is a positive integer index (1, 2, 3, …)

### Particle Type Patterns

| Particle Type | Column Pattern | Examples |
|---------------|----------------|----------|
| **Ions** | `Px_ion{N}`, `Py_ion{N}`, `Pz_ion{N}` | `Px_ion1`, `Py_ion1`, `Pz_ion1` |
| **Neutrals** | `Px_neutral{N}`, `Py_neutral{N}`, `Pz_neutral{N}` | `Px_neutral1`, `Py_neutral1`, `Pz_neutral1` |
| **Neutrals** (single) | `Px_neutral`, `Py_neutral`, `Pz_neutral` | No number needed if only one neutral |
| **Electrons** | `Px_electron{N}`, `Py_electron{N}`, `Pz_electron{N}` | `Px_electron1`, `Py_electron1`, `Pz_electron1` |
| **Generic** | `Px_particle{N}`, `Py_particle{N}`, `Pz_particle{N}` | For particles that don't fit the above categories |

> **Important:** Every particle must have all three components (Px, Py, Pz). Particles with incomplete momentum vectors are discarded.

---

## Example Data Files

### Example 1: D₂O Dissociation (2 ions + 1 neutral + 2 electrons)

This is the standard five-body breakup: D₂O → D⁺ + D⁺ + O + e⁻ + e⁻

```
Px_ion1    Py_ion1    Pz_ion1    Px_ion2    Py_ion2    Pz_ion2    Px_neutral    Py_neutral    Pz_neutral    Px_electron1    Py_electron1    Pz_electron1    Px_electron2    Py_electron2    Pz_electron2
 23.456    -12.789     5.123      -18.234    15.678     -3.456      -5.222       -2.889         -1.667        0.123           -0.456           0.234           0.877            0.467          -0.131
 19.876    -10.234     4.567      -15.432    12.345     -2.890      -4.444       -2.111         -1.677        0.098           -0.345           0.189           0.902            0.389          -0.078
...
```

### Example 2: CO₂ Triple Ionization (3 ions + 3 electrons)

CO₂³⁺ → C⁺ + O⁺ + O⁺ + e⁻ + e⁻ + e⁻

```
Px_ion1,Py_ion1,Pz_ion1,Px_ion2,Py_ion2,Pz_ion2,Px_ion3,Py_ion3,Pz_ion3,Px_electron1,Py_electron1,Pz_electron1,Px_electron2,Py_electron2,Pz_electron2,Px_electron3,Py_electron3,Pz_electron3
45.12,-23.45,12.34,-22.56,11.78,-6.17,-22.56,11.67,-6.17,0.12,-0.34,0.19,0.23,-0.12,0.08,-0.35,0.46,-0.27
...
```

### Example 3: Simple Two-Body (1 ion + 1 electron)

```
Px_ion1 Py_ion1 Pz_ion1 Px_electron1 Py_electron1 Pz_electron1
 34.567  -15.234   8.901     -0.234       0.567      -0.345
 28.123  -11.456   6.789     -0.189       0.432      -0.267
...
```

### Example 4: With Multiple Neutrals

N₂O → N⁺ + N + O + e⁻

```
Px_ion1 Py_ion1 Pz_ion1 Px_neutral1 Py_neutral1 Pz_neutral1 Px_neutral2 Py_neutral2 Pz_neutral2 Px_electron1 Py_electron1 Pz_electron1
 15.234  -8.567   3.456    -7.123       4.234       -1.789       -8.111       4.333       -1.667       0.098       -0.234        0.089
...
```

---

## Momentum Units

All momentum values must be in **atomic units** (a.u.), where the unit of momentum is ℏ/a₀ ≈ 1.993 × 10⁻²⁴ kg·m/s. SCULPT uses the non-relativistic formula E = p²/(2m) internally (with mass in electron masses), and converts to physical display units (eV for energies, degrees for angles) only at plot time.

If your data is in other units, convert to atomic units before uploading:

| From | To atomic units (ℏ/a₀) | Conversion |
|------|------------------------|------------|
| eV/c | a.u. | Multiply by 0.0121 |
| SI (kg·m/s) | a.u. | Divide by 1.993 × 10⁻²⁴ |

> **Tip:** Many COLTRIMS analysis pipelines (e.g., LMF2Root, CoboldPC) can export momentum data directly in atomic units. If your pipeline already outputs in a.u., no conversion is needed.

---

## Particle Configuration Profiles

After uploading your data, SCULPT requires you to create a **configuration profile** that tells the platform about your particles' physical properties. This is done in the **Data Management** tab.

### What a Profile Specifies

For each particle in your data:

| Property | Required For | Description |
|----------|-------------|-------------|
| **Name** | Ions, Neutrals | Species label (e.g., `D+`, `H+`, `O`, `N`) |
| **Mass** | Ions, Neutrals | Mass in atomic mass units (amu) |
| **Charge** | Ions | Charge state (e.g., 1, 2) |

Electron mass is fixed automatically at 1/1836 amu (0.000545 amu).

### Built-in Profiles

SCULPT ships with two pre-configured profiles:

| Profile | Configuration | Particles |
|---------|--------------|-----------|
| **D₂O** | 2 ions + 1 neutral + 2 electrons | D⁺ (2 amu), D⁺ (2 amu), O (16 amu), e⁻, e⁻ |
| **HDO** | 2 ions + 1 neutral + 2 electrons | H⁺ (1 amu), D⁺ (2 amu), O (16 amu), e⁻, e⁻ |

### Creating a Custom Profile

1. In the **Data Management** tab, go to **Molecule Configuration**
2. Set the number of ions, neutrals, and electrons
3. For each ion and neutral, enter the species name and mass (amu)
4. Name the profile and click **Create Profile**
5. Assign the profile to your uploaded file(s)

### Assigning Profiles to Files

Each uploaded file must be assigned a configuration profile before physics features are calculated. This tells SCULPT the particle ordering and masses for your specific data. Go to the **File-Profile Assignments** section and select the appropriate profile for each file.

---

## Computed Physics Features

Once a file is uploaded and assigned a configuration profile, SCULPT automatically computes the following derived quantities:

### Per-Particle Features

| Feature | Formula | Unit | Description |
|---------|---------|------|-------------|
| Momentum magnitude | \|**p**\| = √(Px² + Py² + Pz²) | a.u. | Total momentum |
| Kinetic energy | E = p²/(2m) | a.u. → eV | Particle kinetic energy |
| Polar angle (θ) | θ = arccos(Pz/\|**p**\|) | rad → ° | Angle from z-axis |
| Azimuthal angle (φ) | φ = atan2(Py, Px) | rad → ° | Angle in xy-plane |

### Summary Features

| Feature | Description |
|---------|-------------|
| **KER** | Kinetic Energy Release — sum of all ion kinetic energies |
| **EESum** | Electron Energy Sum — sum of all electron kinetic energies |
| **EESharing** | Electron Energy Sharing — ratio of first electron energy to EESum |
| **TotalEnergy** | Sum of all particle kinetic energies |

### Pairwise Features (for every particle pair i, j)

| Feature | Description |
|---------|-------------|
| Dot product | **p**ᵢ · **p**ⱼ |
| Cosine similarity | (**p**ᵢ · **p**ⱼ) / (\|**p**ᵢ\| · \|**p**ⱼ\|) |
| Relative angle | arccos(cosine similarity) |
| Momentum difference | \| \|**p**ᵢ\| - \|**p**ⱼ\| \| |
| Δφ | Azimuthal angle difference (with periodicity) |
| Δθ | Polar angle difference |
| φ ratio | φᵢ / φⱼ |
| θ ratio | θᵢ / θⱼ |

> **Note:** The number of pairwise features scales as N(N-1)/2, where N is the total number of particles. A 5-particle system produces 10 pairs; a 6-particle system produces 15.

---

## Re-uploading Saved Selections

SCULPT can export subsets of your data (lasso/box selections, filtered data) as CSV files. These exported files can be re-uploaded into SCULPT as **selection files**.

### What Selection Files Contain

Exported selection files include:

- **Full momentum data** — all particle momentum columns (Px, Py, Pz) converted back to their original column names, allowing physics features to be recalculated on re-upload
- **UMAP coordinates** — `UMAP1` and `UMAP2` embedding coordinates for overlay on UMAP plots
- **Source identifier** — `file_label` column indicating the original source file

This means re-uploaded selections behave like regular data files with the added benefit of carrying their UMAP coordinates. You can assign a configuration profile to a re-uploaded selection and have all physics features (KER, angles, pairwise quantities, etc.) recomputed from the preserved momentum data.

### How Selection Files Are Detected

When a file is uploaded, SCULPT checks whether the dataframe contains **all three** of these columns:

| Column | Description |
|--------|-------------|
| `UMAP1` | First UMAP embedding coordinate |
| `UMAP2` | Second UMAP embedding coordinate |
| `file_label` | Source file identifier |

If all three are present, the file is classified as a saved selection and can be overlaid on UMAP plots alongside your original data. Since the momentum data is preserved, you can also assign a configuration profile and use the selection for further physics analysis.

If any of these three columns is missing, the file is treated as a regular COLTRIMS data file and must follow the standard naming conventions described above.

> **Note:** Avoid naming your own data columns `UMAP1`, `UMAP2`, or `file_label`, as this will cause SCULPT to misclassify your file as a saved selection.

---

## Troubleshooting

### "No valid particle columns found"
Your column names don't match any recognized pattern. Check that:
- Column names follow the `P{x/y/z}_{type}{number}` format exactly
- There are no extra spaces in column names
- Each particle has all three components (Px, Py, Pz)

### "No complete particle momentum vectors found"
Some particles have only 1 or 2 of the 3 required momentum components. Ensure every particle has Px, Py, and Pz columns.

### Features not appearing in dropdowns
Make sure you have:
1. Created a configuration profile in **Data Management**
2. Assigned the profile to your file(s)
3. Run UMAP or generated a plot (features are calculated at analysis time)

### Data looks wrong after upload
- Verify your momentum values are in **atomic units** (see [Momentum Units](#momentum-units) above)
- Check that `ion1` in your file header corresponds to the first ion in your configuration profile, `ion2` to the second, etc. — the numbering must be consistent between the file and the profile
- Ensure the header row has no trailing whitespace or special characters

---

## Quick Start Checklist

1. ☐ Prepare your data file with the correct column naming convention
2. ☐ Ensure all momentum values are in atomic units
3. ☐ Upload the file via the **Data Management** tab
4. ☐ Create a configuration profile specifying particle types, masses, and charges
5. ☐ Assign the profile to your uploaded file
6. ☐ Navigate to **Basic Analysis** → select features → **Run UMAP**
