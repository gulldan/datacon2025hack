#!/usr/bin/env python3
"""GPU-Accelerated Molecular Docking Script
Uses AutoDock-GPU and Vina-GPU for high-performance docking
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger


@dataclass
class GPUDockingConfig:
    """Configuration for GPU docking"""

    use_gpu: bool = True
    gpu_device: int = 0
    autodock_gpu_path: str = "/home/qwerty/github/datacon2025hack/gpu_docking_tools/AutoDock-GPU-develop/bin/autodock_gpu_128wi"
    vina_gpu_path: str = (
        "/home/qwerty/github/datacon2025hack/gpu_docking_tools/Vina-GPU-2.1-main/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1"
    )
    batch_size: int = 100
    max_concurrent_jobs: int = 4
    nrun: int = 10
    exhaustiveness: int = 32
    num_modes: int = 20
    energy_range: float = 4.0


class GPUDockingEngine:
    """GPU-accelerated molecular docking engine"""

    def __init__(self, config: GPUDockingConfig):
        self.config = config
        self.logger = setup_logger()
        self.setup_gpu_environment()

    def setup_gpu_environment(self):
        """Setup GPU environment and verify GPU availability"""
        try:
            # Check NVIDIA GPU
            result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning("NVIDIA GPU not available, falling back to CPU")
                self.config.use_gpu = False
                return

            # Check AutoDock-GPU availability
            if not os.path.exists(self.config.autodock_gpu_path):
                self.logger.warning(f"AutoDock-GPU not found at {self.config.autodock_gpu_path}")
                self.config.use_gpu = False
                return

            # Test AutoDock-GPU
            result = subprocess.run(
                [self.config.autodock_gpu_path, "--help"], check=False, capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                self.logger.warning("AutoDock-GPU not working properly")
                self.config.use_gpu = False
                return

            self.logger.info("GPU environment setup successful")
            self.logger.info(f"Using GPU device: {self.config.gpu_device}")

        except Exception as e:
            self.logger.error(f"GPU setup failed: {e}")
            self.config.use_gpu = False

    def prepare_ligand_for_autodock_gpu(self, ligand_path: str, output_dir: str) -> str:
        """Prepare ligand for AutoDock-GPU (convert to PDBQT if needed)"""
        ligand_name = Path(ligand_path).stem
        pdbqt_path = os.path.join(output_dir, f"{ligand_name}.pdbqt")

        if ligand_path.endswith(".pdbqt"):
            # Already in PDBQT format
            if ligand_path != pdbqt_path:
                subprocess.run(["cp", ligand_path, pdbqt_path], check=True)
        else:
            # Convert to PDBQT using obabel
            cmd = ["obabel", ligand_path, "-O", pdbqt_path, "--gen3d", "--minimize", "--ff", "UFF"]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to convert ligand: {result.stderr}")

        return pdbqt_path

    def prepare_receptor_maps(
        self, receptor_path: str, output_dir: str, center: tuple[float, float, float], size: tuple[float, float, float]
    ) -> str:
        """Prepare receptor maps for AutoDock-GPU"""
        receptor_name = Path(receptor_path).stem
        maps_dir = os.path.join(output_dir, f"{receptor_name}_maps")
        os.makedirs(maps_dir, exist_ok=True)

        # Create GPF file
        gpf_path = os.path.join(maps_dir, f"{receptor_name}.gpf")
        fld_path = os.path.join(maps_dir, f"{receptor_name}.maps.fld")

        # Generate grid parameter file
        gpf_content = f"""npts {int(size[0] / 0.375)} {int(size[1] / 0.375)} {int(size[2] / 0.375)}
gridfld {receptor_name}.maps.fld
spacing 0.375
receptor_types A C HD N OA SA
ligand_types A C HD N OA SA
receptor {receptor_path}
gridcenter {center[0]} {center[1]} {center[2]}
smooth 0.5
map {receptor_name}.A.map
map {receptor_name}.C.map
map {receptor_name}.HD.map
map {receptor_name}.N.map
map {receptor_name}.OA.map
map {receptor_name}.SA.map
elecmap {receptor_name}.e.map
dsolvmap {receptor_name}.d.map
dielectric -0.1465
"""

        with open(gpf_path, "w") as f:
            f.write(gpf_content)

        # Generate maps using autogrid
        cmd = ["autogrid4", "-p", gpf_path, "-l", os.path.join(maps_dir, f"{receptor_name}.glg")]
        result = subprocess.run(cmd, check=False, cwd=maps_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate maps: {result.stderr}")

        return fld_path

    def dock_ligand_autodock_gpu(self, ligand_path: str, fld_path: str, output_dir: str) -> dict:
        """Dock single ligand using AutoDock-GPU"""
        ligand_name = Path(ligand_path).stem
        output_path = os.path.join(output_dir, f"{ligand_name}_gpu.dlg")

        cmd = [
            self.config.autodock_gpu_path,
            "--lfile",
            ligand_path,
            "--ffile",
            fld_path,
            "--resnam",
            ligand_name,
            "--nrun",
            str(self.config.nrun),
            "--devnum",
            str(self.config.gpu_device + 1),  # AutoDock-GPU uses 1-based indexing
            "--dlgoutput",
            "1",
            "--xmloutput",
            "0",
        ]

        start_time = time.time()
        result = subprocess.run(cmd, check=False, cwd=output_dir, capture_output=True, text=True)
        end_time = time.time()

        if result.returncode != 0:
            self.logger.error(f"AutoDock-GPU failed for {ligand_name}: {result.stderr}")
            return {
                "ligand": ligand_name,
                "binding_affinity": 0.0,
                "rmsd": 0.0,
                "success": False,
                "time": end_time - start_time,
                "error": result.stderr,
            }

        # Parse results
        binding_affinity, rmsd = self.parse_autodock_gpu_results(output_path)

        return {
            "ligand": ligand_name,
            "binding_affinity": binding_affinity,
            "rmsd": rmsd,
            "success": True,
            "time": end_time - start_time,
            "error": None,
        }

    def parse_autodock_gpu_results(self, dlg_path: str) -> tuple[float, float]:
        """Parse AutoDock-GPU DLG file to extract binding affinity and RMSD"""
        try:
            with open(dlg_path) as f:
                lines = f.readlines()

            binding_affinity = 0.0
            rmsd = 0.0

            for line in lines:
                if line.startswith("DOCKED: USER    Final Intermolecular Energy"):
                    # Extract binding affinity
                    parts = line.split()
                    if len(parts) >= 8:
                        binding_affinity = float(parts[7])
                elif line.startswith("DOCKED: USER    Final Total Internal Energy"):
                    # Extract RMSD if available
                    continue
                elif "RMSD from reference structure" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "RMSD" and i + 4 < len(parts):
                            rmsd = float(parts[i + 4])
                            break

            return binding_affinity, rmsd

        except Exception as e:
            self.logger.error(f"Failed to parse {dlg_path}: {e}")
            return 0.0, 0.0

    def dock_ligand_vina_gpu(
        self,
        ligand_path: str,
        receptor_path: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        output_dir: str,
    ) -> dict:
        """Dock single ligand using Vina-GPU"""
        ligand_name = Path(ligand_path).stem
        output_path = os.path.join(output_dir, f"{ligand_name}_vina_gpu.pdbqt")

        # Create config file
        config_path = os.path.join(output_dir, f"{ligand_name}_config.txt")
        config_content = f"""receptor = {receptor_path}
ligand = {ligand_path}
opencl_binary_path = {Path(self.config.vina_gpu_path).parent}
output = {output_path}
log = {os.path.join(output_dir, f"{ligand_name}_vina_gpu.log")}
thread = 8000
center_x = {center[0]}
center_y = {center[1]}
center_z = {center[2]}
size_x = {size[0]}
size_y = {size[1]}
size_z = {size[2]}
exhaustiveness = {self.config.exhaustiveness}
num_modes = {self.config.num_modes}
energy_range = {self.config.energy_range}
"""

        with open(config_path, "w") as f:
            f.write(config_content)

        cmd = [self.config.vina_gpu_path, "--config", config_path]

        start_time = time.time()
        result = subprocess.run(cmd, check=False, cwd=output_dir, capture_output=True, text=True)
        end_time = time.time()

        if result.returncode != 0:
            self.logger.error(f"Vina-GPU failed for {ligand_name}: {result.stderr}")
            return {
                "ligand": ligand_name,
                "binding_affinity": 0.0,
                "rmsd": 0.0,
                "success": False,
                "time": end_time - start_time,
                "error": result.stderr,
            }

        # Parse results
        binding_affinity, rmsd = self.parse_vina_gpu_results(output_path)

        return {
            "ligand": ligand_name,
            "binding_affinity": binding_affinity,
            "rmsd": rmsd,
            "success": True,
            "time": end_time - start_time,
            "error": None,
        }

    def parse_vina_gpu_results(self, pdbqt_path: str) -> tuple[float, float]:
        """Parse Vina-GPU PDBQT output to extract binding affinity"""
        try:
            with open(pdbqt_path) as f:
                lines = f.readlines()

            binding_affinity = 0.0

            for line in lines:
                if line.startswith("REMARK VINA RESULT:"):
                    # Extract binding affinity from VINA RESULT line
                    parts = line.split()
                    if len(parts) >= 4:
                        binding_affinity = float(parts[3])
                        break

            return binding_affinity, 0.0  # RMSD not available in Vina output

        except Exception as e:
            self.logger.error(f"Failed to parse {pdbqt_path}: {e}")
            return 0.0, 0.0

    def run_gpu_docking_batch(
        self,
        ligand_files: list[str],
        receptor_path: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        output_dir: str,
        engine: str = "autodock_gpu",
    ) -> list[dict]:
        """Run GPU docking on a batch of ligands"""
        os.makedirs(output_dir, exist_ok=True)
        results = []

        prepared_ligands = []
        dock_func = None

        if engine == "autodock_gpu":
            # Prepare receptor maps once
            fld_path = self.prepare_receptor_maps(receptor_path, output_dir, center, size)

            # Prepare ligands
            for ligand_file in ligand_files:
                try:
                    pdbqt_path = self.prepare_ligand_for_autodock_gpu(ligand_file, output_dir)
                    prepared_ligands.append(pdbqt_path)
                except Exception as e:
                    self.logger.error(f"Failed to prepare ligand {ligand_file}: {e}")
                    results.append(
                        {
                            "ligand": Path(ligand_file).stem,
                            "binding_affinity": 0.0,
                            "rmsd": 0.0,
                            "success": False,
                            "time": 0.0,
                            "error": str(e),
                        }
                    )

            # Dock ligands using GPU
            dock_func = partial(self.dock_ligand_autodock_gpu, fld_path=fld_path, output_dir=output_dir)

        elif engine == "vina_gpu":
            # Prepare docking function
            dock_func = partial(
                self.dock_ligand_vina_gpu, receptor_path=receptor_path, center=center, size=size, output_dir=output_dir
            )
            prepared_ligands = ligand_files

        if dock_func is None:
            raise ValueError(f"Unknown docking engine: {engine}")

        # Run docking with progress bar
        with tqdm(total=len(prepared_ligands), desc=f"GPU Docking ({engine})") as pbar:
            if self.config.use_gpu:
                # GPU can handle multiple ligands efficiently
                for ligand_path in prepared_ligands:
                    result = dock_func(ligand_path)
                    results.append(result)
                    pbar.update(1)
            else:
                # Fallback to CPU parallelization
                with ProcessPoolExecutor(max_workers=self.config.max_concurrent_jobs) as executor:
                    future_to_ligand = {executor.submit(dock_func, ligand_path): ligand_path for ligand_path in prepared_ligands}

                    for future in as_completed(future_to_ligand):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)

        return results


def main():
    """Main function for GPU docking"""
    import argparse

    parser = argparse.ArgumentParser(description="GPU-Accelerated Molecular Docking")
    parser.add_argument("--ligand_dir", required=True, help="Directory containing ligand files")
    parser.add_argument("--receptor", required=True, help="Receptor PDB file")
    parser.add_argument("--center", nargs=3, type=float, required=True, help="Binding site center coordinates (x y z)")
    parser.add_argument("--size", nargs=3, type=float, default=[20.0, 20.0, 20.0], help="Binding site size (x y z)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--engine", choices=["autodock_gpu", "vina_gpu"], default="autodock_gpu", help="Docking engine to use")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--nrun", type=int, default=10, help="Number of runs per ligand")

    args = parser.parse_args()

    # Setup configuration
    config = GPUDockingConfig(batch_size=args.batch_size, nrun=args.nrun)

    # Initialize GPU docking engine
    engine = GPUDockingEngine(config)

    # Get ligand files
    ligand_dir = Path(args.ligand_dir)
    ligand_files = []
    for ext in ["*.pdb", "*.sdf", "*.mol2", "*.pdbqt"]:
        ligand_files.extend(ligand_dir.glob(ext))

    if not ligand_files:
        print(f"No ligand files found in {ligand_dir}")
        return

    print(f"Found {len(ligand_files)} ligand files")
    print(f"Using {args.engine} engine")
    print(f"GPU enabled: {config.use_gpu}")

    # Run docking
    start_time = time.time()
    results = engine.run_gpu_docking_batch(
        ligand_files=[str(f) for f in ligand_files],
        receptor_path=args.receptor,
        center=tuple(args.center),
        size=tuple(args.size),
        output_dir=args.output_dir,
        engine=args.engine,
    )
    end_time = time.time()

    # Save results
    df = pd.DataFrame(results)
    results_file = os.path.join(args.output_dir, "gpu_docking_results.csv")
    df.to_csv(results_file, index=False)

    # Print summary
    successful = df[df["success"] == True]
    total_time = end_time - start_time

    print("\n=== GPU Docking Summary ===")
    print(f"Total ligands: {len(ligand_files)}")
    print(f"Successful dockings: {len(successful)}")
    print(f"Failed dockings: {len(df) - len(successful)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per ligand: {total_time / len(ligand_files):.2f} seconds")
    print(f"Throughput: {len(ligand_files) / total_time:.2f} ligands/second")

    if len(successful) > 0:
        print(f"Best binding affinity: {successful['binding_affinity'].min():.2f} kcal/mol")
        print(f"Average binding affinity: {successful['binding_affinity'].mean():.2f} kcal/mol")

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
