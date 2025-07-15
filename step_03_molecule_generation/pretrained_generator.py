"""Pretrained molecular generator using Hugging Face models.
Uses entropy/gpt2_zinc_87m as base model for SMILES generation.
"""

import re

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils.logger import LOGGER


class PretrainedMolecularGenerator:
    """Molecular generator using pre-trained Hugging Face models.
    Uses entropy/gpt2_zinc_87m as base model.
    """

    def __init__(self, model_name: str = "entropy/gpt2_zinc_87m", max_length: int = 256, device: str = "auto"):
        """Initialize the pretrained molecular generator.

        Args:
            model_name: Hugging Face model name
            max_length: Maximum sequence length
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._setup_device(device)
        self.logger = LOGGER

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _setup_device(self, device: str) -> str:
        """Setup device for inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")

            # Load tokenizer
            self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name, max_len=self.max_length)

            # Load model
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def generate_molecules(
        self,
        num_molecules: int = 1000,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        batch_size: int = 32,
        filter_valid: bool = True,
    ) -> list[str]:
        """Generate molecules using the pre-trained model.

        Args:
            num_molecules: Number of molecules to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            batch_size: Batch size for generation
            filter_valid: Whether to filter valid molecules

        Returns:
            List of generated SMILES strings
        """
        generated_smiles = []

        with torch.no_grad():
            for i in range(0, num_molecules, batch_size):
                current_batch_size = min(batch_size, num_molecules - i)

                # Create input tensors
                inputs = torch.tensor([[self.tokenizer.bos_token_id]] * current_batch_size)
                inputs = inputs.to(self.device)

                # Generate
                outputs = self.model.generate(
                    inputs,
                    do_sample=True,
                    max_length=self.max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=1,
                )

                # Decode outputs
                batch_smiles = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Clean and validate
                for smiles in batch_smiles:
                    smiles = self._clean_smiles(smiles)
                    if smiles and (not filter_valid or self._is_valid_smiles(smiles)):
                        generated_smiles.append(smiles)

                if i % (batch_size * 10) == 0:
                    self.logger.info(f"Generated {len(generated_smiles)} valid molecules so far...")

        self.logger.info(f"Generated {len(generated_smiles)} valid molecules total")
        return generated_smiles

    def _clean_smiles(self, smiles: str) -> str:
        """Clean generated SMILES string."""
        # Remove any special tokens that might remain
        smiles = re.sub(r"<[^>]*>", "", smiles)
        smiles = smiles.strip()

        # Basic SMILES cleaning
        if not smiles or len(smiles) < 3:
            return ""

        return smiles

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is valid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def compute_embeddings(self, smiles_list: list[str]) -> np.ndarray:
        """Compute embeddings for SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Numpy array of embeddings
        """
        embeddings = []

        with torch.no_grad():
            for smiles in smiles_list:
                # Tokenize
                inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get hidden states
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # Last layer

                # Mean pooling
                attention_mask = inputs["attention_mask"]
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                embedding = masked_hidden.sum(1) / attention_mask.sum(-1).unsqueeze(-1)

                embeddings.append(embedding.cpu().numpy())

        return np.vstack(embeddings)

    def evaluate_molecules(self, smiles_list: list[str]) -> dict[str, float]:
        """Evaluate generated molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary of evaluation metrics
        """
        valid_count = 0
        unique_smiles = set()
        qed_scores = []
        mw_scores = []
        logp_scores = []

        for smiles in smiles_list:
            if self._is_valid_smiles(smiles):
                valid_count += 1
                unique_smiles.add(smiles)

                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Calculate properties
                    qed_scores.append(QED.qed(mol))
                    mw_scores.append(Descriptors.MolWt(mol))
                    logp_scores.append(Descriptors.MolLogP(mol))

        metrics = {
            "validity": valid_count / len(smiles_list) if smiles_list else 0,
            "uniqueness": len(unique_smiles) / len(smiles_list) if smiles_list else 0,
            "diversity": len(unique_smiles) / valid_count if valid_count > 0 else 0,
            "mean_qed": np.mean(qed_scores) if qed_scores else 0,
            "mean_mw": np.mean(mw_scores) if mw_scores else 0,
            "mean_logp": np.mean(logp_scores) if logp_scores else 0,
            "total_generated": len(smiles_list),
            "valid_molecules": valid_count,
            "unique_molecules": len(unique_smiles),
        }

        return metrics

    def fine_tune_for_target(self, target_smiles: list[str], learning_rate: float = 1e-5, epochs: int = 3, batch_size: int = 16):
        """Fine-tune the model on target-specific SMILES.

        Args:
            target_smiles: List of target SMILES for fine-tuning
            learning_rate: Learning rate for fine-tuning
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.logger.info(f"Fine-tuning model on {len(target_smiles)} target molecules")

        # Prepare data
        from torch.utils.data import DataLoader, Dataset

        class SMILESDataset(Dataset):
            def __init__(self, smiles_list, tokenizer, max_length):
                self.smiles_list = smiles_list
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.smiles_list)

            def __getitem__(self, idx):
                smiles = self.smiles_list[idx]
                # Add BOS and EOS tokens
                smiles = f"{self.tokenizer.bos_token}{smiles}{self.tokenizer.eos_token}"

                encoding = self.tokenizer(
                    smiles, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
                )

                return {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "labels": encoding["input_ids"].flatten(),
                }

        # Create dataset and dataloader
        dataset = SMILESDataset(target_smiles, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                # Explicitly pass labels to avoid loss_type warning
                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.model.eval()
        self.logger.info("Fine-tuning completed")


def main():
    """Test the pretrained generator."""
    generator = PretrainedMolecularGenerator()

    # Generate molecules
    molecules = generator.generate_molecules(num_molecules=100, temperature=1.0, batch_size=16)

    # Evaluate
    metrics = generator.evaluate_molecules(molecules)

    print("Generated molecules:")
    for i, smiles in enumerate(molecules[:10]):
        print(f"{i + 1}: {smiles}")

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
