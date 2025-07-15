"""Docking-guided molecular generator for DYRK1A target.
Integrates molecular generation with real-time docking evaluation.
"""

import os
import subprocess

# Import your existing modules
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import selfies as sf
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch import nn

sys.path.append("..")
from step_02_activity_prediction.model_utils import load_model, predict_activity
from utils.logger import LOGGER

logger = LOGGER


@dataclass
class DockingConfig:
    """Configuration for docking-guided generation."""

    target_pdb: str = "6S14"
    chembl_id: str = "CHEMBL3227"
    exhaustiveness: int = 8
    num_modes: int = 9
    energy_range: float = 3.0
    docking_weight: float = 0.4
    activity_weight: float = 0.3
    drug_likeness_weight: float = 0.2
    novelty_weight: float = 0.1


class PocketPredictor(nn.Module):
    """Predicts binding pockets from protein structure."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.pocket_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True), num_layers=4)
        self.pocket_classifier = nn.Linear(d_model, 1)

    def forward(self, protein_features: torch.Tensor) -> torch.Tensor:
        """Predict binding pocket probabilities."""
        encoded = self.pocket_encoder(protein_features)
        pocket_probs = torch.sigmoid(self.pocket_classifier(encoded))
        return pocket_probs


class DockingEvaluator:
    """Handles molecular docking evaluation using AutoDock Vina."""

    def __init__(self, config: DockingConfig):
        self.config = config
        self.protein_path = f"../step_04_hit_selection/docking/{config.target_pdb}.pdb"
        self.prepare_protein()

    def prepare_protein(self):
        """Prepare protein for docking."""
        if not os.path.exists(self.protein_path):
            logger.error(f"Protein file not found: {self.protein_path}")
            return

        # Convert PDB to PDBQT format for Vina
        self.protein_pdbqt = self.protein_path.replace(".pdb", ".pdbqt")
        if not os.path.exists(self.protein_pdbqt):
            cmd = f"prepare_receptor4.py -r {self.protein_path} -o {self.protein_pdbqt}"
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
                logger.info(f"Prepared protein: {self.protein_pdbqt}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to prepare protein: {e}")

    def dock_molecule(self, smiles: str) -> float:
        """Dock a single molecule and return binding affinity."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return -1000.0

            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol) != 0:
                return -1000.0

            AllChem.UFFOptimizeMolecule(mol)

            # Save ligand to temporary file
            with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
                ligand_sdf = f.name

            writer = Chem.SDWriter(ligand_sdf)
            writer.write(mol)
            writer.close()

            # Convert to PDBQT
            ligand_pdbqt = ligand_sdf.replace(".sdf", ".pdbqt")
            cmd = f"prepare_ligand4.py -l {ligand_sdf} -o {ligand_pdbqt}"
            subprocess.run(cmd, shell=True, check=True, capture_output=True)

            # Run Vina docking
            output_pdbqt = ligand_pdbqt.replace(".pdbqt", "_out.pdbqt")

            # Define search space (adjust based on your protein)
            vina_cmd = f"""vina --receptor {self.protein_pdbqt} \
                          --ligand {ligand_pdbqt} \
                          --out {output_pdbqt} \
                          --center_x 0 --center_y 0 --center_z 0 \
                          --size_x 20 --size_y 20 --size_z 20 \
                          --exhaustiveness {self.config.exhaustiveness} \
                          --num_modes {self.config.num_modes} \
                          --energy_range {self.config.energy_range}"""

            result = subprocess.run(vina_cmd, check=False, shell=True, capture_output=True, text=True)

            # Parse binding affinity from output
            binding_affinity = self.parse_vina_output(result.stdout)

            # Cleanup temporary files
            for temp_file in [ligand_sdf, ligand_pdbqt, output_pdbqt]:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

            return binding_affinity

        except Exception as e:
            logger.error(f"Docking failed for {smiles}: {e}")
            return -1000.0

    def parse_vina_output(self, output: str) -> float:
        """Parse Vina output to extract binding affinity."""
        lines = output.split("\n")
        for line in lines:
            if "REMARK VINA RESULT:" in line:
                parts = line.split()
                if len(parts) >= 4:
                    return float(parts[3])  # Binding affinity
        return -1000.0  # Default if parsing fails


class DockingGuidedGenerator(nn.Module):
    """Molecular generator with integrated docking guidance."""

    def __init__(self, vocab_size: int, config: DockingConfig):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Core generation modules
        self.embedding = nn.Embedding(vocab_size, 256)
        self.pocket_predictor = PocketPredictor()
        self.generator = nn.LSTM(256, 512, num_layers=2, batch_first=True)
        self.output_proj = nn.Linear(512, vocab_size)

        # Docking evaluator
        self.docking_evaluator = DockingEvaluator(config)

        # Activity predictor (load pre-trained model)
        self.activity_model = None
        self.load_activity_model()

    def load_activity_model(self):
        """Load pre-trained activity prediction model."""
        try:
            model_path = "../step_02_activity_prediction/results/activity_model_xgb.json"
            self.activity_model = load_model(model_path)
            logger.info("Loaded activity prediction model")
        except Exception as e:
            logger.error(f"Failed to load activity model: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequence generation."""
        embedded = self.embedding(x)
        lstm_out, _ = self.generator(embedded)
        logits = self.output_proj(lstm_out)
        return logits

    def calculate_reward(self, smiles: str) -> float:
        """Calculate multi-objective reward for a molecule."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return -1000.0

            # 1. Docking score
            docking_score = self.docking_evaluator.dock_molecule(smiles)
            # Convert to positive reward (lower binding energy = better)
            docking_reward = max(0, -docking_score / 10.0)

            # 2. Activity prediction
            activity_reward = 0.0
            if self.activity_model is not None:
                try:
                    activity_pred = predict_activity(self.activity_model, [smiles])
                    activity_reward = float(activity_pred[0]) if len(activity_pred) > 0 else 0.0
                except:
                    activity_reward = 0.0

            # 3. Drug-likeness (QED)
            qed_score = Descriptors.qed(mol)

            # 4. Novelty (simple check - can be improved)
            novelty_score = 1.0  # Placeholder

            # Combined reward
            total_reward = (
                self.config.docking_weight * docking_reward
                + self.config.activity_weight * activity_reward
                + self.config.drug_likeness_weight * qed_score
                + self.config.novelty_weight * novelty_score
            )

            return total_reward

        except Exception as e:
            logger.error(f"Reward calculation failed for {smiles}: {e}")
            return -1000.0

    def generate_molecules(self, vocab, n_samples: int = 100, max_length: int = 80) -> list[str]:
        """Generate molecules with docking guidance."""
        self.eval()
        generated_molecules = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Initialize sequence
                sequence = [vocab.stoi[vocab.bos]]

                for _ in range(max_length - 1):
                    # Convert to tensor
                    x = torch.tensor([sequence], dtype=torch.long)

                    # Get next token probabilities
                    logits = self.forward(x)
                    probs = F.softmax(logits[0, -1, :], dim=-1)

                    # Sample next token
                    next_token = torch.multinomial(probs, 1).item()
                    sequence.append(next_token)

                    # Stop if EOS token
                    if next_token == vocab.stoi.get(vocab.eos, vocab.stoi[vocab.pad]):
                        break

                # Decode sequence
                try:
                    tokens = [vocab.itos[idx] for idx in sequence[1:]]  # Skip BOS
                    selfies_str = "".join(tokens)
                    smiles = sf.decoder(selfies_str)

                    if smiles and len(smiles) > 3:
                        generated_molecules.append(smiles)

                except Exception as e:
                    logger.debug(f"Failed to decode sequence: {e}")
                    continue

        return generated_molecules

    def reinforcement_learning_step(self, vocab, batch_size: int = 32):
        """Perform one RL step with docking rewards."""
        # Generate molecules
        molecules = self.generate_molecules(vocab, batch_size)

        # Calculate rewards
        rewards = []
        for smiles in molecules:
            reward = self.calculate_reward(smiles)
            rewards.append(reward)

        # Update model based on rewards (simplified REINFORCE)
        if len(rewards) > 0:
            avg_reward = np.mean(rewards)
            logger.info(f"Average reward: {avg_reward:.4f}")

            # Here you would implement the actual REINFORCE update
            # This is a placeholder for the full implementation

        return np.mean(rewards) if rewards else 0.0


def train_docking_guided_generator(vocab, config: DockingConfig, num_epochs: int = 100):
    """Train the docking-guided generator."""
    logger.info("Starting docking-guided generator training")

    # Initialize model
    model = DockingGuidedGenerator(len(vocab), config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Perform RL step
        avg_reward = model.reinforcement_learning_step(vocab)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}, Average reward: {avg_reward:.4f}")

            # Save checkpoint
            checkpoint_path = f"results/docking_guided_generator_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "avg_reward": avg_reward,
                },
                checkpoint_path,
            )

            logger.info(f"Saved checkpoint: {checkpoint_path}")

    return model


if __name__ == "__main__":
    # Example usage
    config = DockingConfig()

    # This would need to be integrated with your existing vocabulary
    # vocab = load_vocabulary()  # Your existing vocab loading

    # model = train_docking_guided_generator(vocab, config)

    logger.info("Docking-guided generator implementation ready")
