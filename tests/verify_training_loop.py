"""Verify the complete training pipeline for DNN and GCNN models.

This script tests the entire pipeline:
1. Data loading with OPFDataModule
2. Model initialization (AdmittanceDNN and GCNN)
3. OPFTask setup with physics loss
4. PyTorch Lightning training loop

It runs one optimization step for each model to confirm the entire
pipeline (Data -> Model -> Task -> Loss) is connected correctly.

Usage:
    python tests/verify_training_loop.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from deep_opf.data import OPFDataModule
from deep_opf.models import AdmittanceDNN, GCNN
from deep_opf.task import OPFTask


def verify_dnn_pipeline():
    """Verify DNN training pipeline."""
    print("=" * 60)
    print("Verifying DNN (AdmittanceDNN) Training Pipeline")
    print("=" * 60)
    
    # 1. Initialize DataModule with 'flat' feature type
    data_dir = Path("legacy/gcnn_opf_01/data_matlab_npz")
    print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Please ensure you have the dataset files in the legacy folder."
        )
    
    dm = OPFDataModule(
        data_dir=data_dir,
        batch_size=4,  # Small batch for testing
        feature_type="flat",
        num_workers=0,  # Windows compatibility
        normalize=True,
    )
    
    # 2. Setup the DataModule
    dm.setup(stage="fit")
    
    # Get dataset statistics for model initialization
    n_bus = dm.n_bus
    n_gen = dm.n_gen
    input_dim = dm.input_dim
    
    print(f"System dimensions: n_bus={n_bus}, n_gen={n_gen}")
    print(f"DNN input dimension: {input_dim}")
    
    # 3. Initialize AdmittanceDNN
    model = AdmittanceDNN(
        input_dim=input_dim,
        hidden_dim=32,  # Small for testing
        num_layers=2,
        n_gen=n_gen,
        n_bus=n_bus,
        dropout=0.1,
    )
    
    print(f"DNN model created: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Get generator bus indices from dataset
    # Load from topology_operators.npz to get actual generator bus mapping
    import numpy as np
    topo_data = np.load(data_dir / "topology_operators.npz")
    gen_bus_indices = topo_data["gen_bus_map"].tolist()
    print(f"Generator bus indices from data: {gen_bus_indices}")
    
    # 5. Initialize OPFTask
    task = OPFTask(
        model=model,
        lr=1e-3,
        kappa=0.1,  # Physics loss weight
        weight_decay=1e-4,
        gen_bus_indices=gen_bus_indices,
        n_bus=n_bus,
    )
    
    print(f"OPFTask created with physics loss weight kappa={task.kappa}")
    
    # 6. Run fast development training
    print("\nRunning fast development training (1 batch)...")
    trainer = pl.Trainer(
        fast_dev_run=True,  # Run only 1 batch
        logger=CSVLogger(save_dir="tests/logs", name="dnn_test"),
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    try:
        trainer.fit(task, datamodule=dm)
        print("✅ DNN training pipeline verification PASSED")
        return True
    except Exception as e:
        print(f"❌ DNN training pipeline verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_gcnn_pipeline():
    """Verify GCNN training pipeline."""
    print("\n" + "=" * 60)
    print("Verifying GCNN Training Pipeline")
    print("=" * 60)
    
    # 1. Initialize DataModule with 'graph' feature type
    data_dir = Path("legacy/gcnn_opf_01/data_matlab_npz")
    print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Please ensure you have the dataset files in the legacy folder."
        )
    
    dm = OPFDataModule(
        data_dir=data_dir,
        batch_size=4,  # Small batch for testing
        feature_type="graph",
        feature_params={"feature_iterations": 5},  # Use 5 iterations
        num_workers=0,  # Windows compatibility
        normalize=True,
    )
    
    # 2. Setup the DataModule
    dm.setup(stage="fit")
    
    # Get dataset statistics for model initialization
    n_bus = dm.n_bus
    n_gen = dm.n_gen
    feature_iterations = dm.feature_iterations
    
    print(f"System dimensions: n_bus={n_bus}, n_gen={n_gen}")
    print(f"GCNN feature iterations: {feature_iterations}")
    
    # 3. Initialize GCNN
    model = GCNN(
        n_bus=n_bus,
        n_gen=n_gen,
        in_channels=feature_iterations,  # e_0_k and f_0_k have k channels
        hidden_channels=8,  # Small for testing
        n_layers=2,
        fc_hidden_dim=32,  # Small for testing
        n_fc_layers=2,
        dropout=0.1,
    )
    
    print(f"GCNN model created: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Get generator bus indices from dataset
    # Load from topology_operators.npz to get actual generator bus mapping
    import numpy as np
    topo_data = np.load(data_dir / "topology_operators.npz")
    gen_bus_indices = topo_data["gen_bus_map"].tolist()
    print(f"Generator bus indices from data: {gen_bus_indices}")
    
    # 5. Initialize OPFTask
    task = OPFTask(
        model=model,
        lr=1e-3,
        kappa=0.1,  # Physics loss weight
        weight_decay=1e-4,
        gen_bus_indices=gen_bus_indices,
        n_bus=n_bus,
    )
    
    print(f"OPFTask created with physics loss weight kappa={task.kappa}")
    
    # 6. Run fast development training
    print("\nRunning fast development training (1 batch)...")
    trainer = pl.Trainer(
        fast_dev_run=True,  # Run only 1 batch
        logger=CSVLogger(save_dir="tests/logs", name="gcnn_test"),
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    try:
        trainer.fit(task, datamodule=dm)
        print("✅ GCNN training pipeline verification PASSED")
        return True
    except Exception as e:
        print(f"❌ GCNN training pipeline verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_data_loading():
    """Quick verification of data loading."""
    print("=" * 60)
    print("Verifying Data Loading")
    print("=" * 60)
    
    data_dir = Path("legacy/gcnn_opf_01/data_matlab_npz")
    
    # Test DNN data loading
    print("\nTesting DNN (flat) data loading...")
    dm_dnn = OPFDataModule(
        data_dir=data_dir,
        batch_size=2,
        feature_type="flat",
        normalize=True,
    )
    dm_dnn.setup(stage="fit")
    
    train_loader = dm_dnn.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Input shape: {batch['input'].shape}")
    print(f"  PG label shape: {batch['pg_label'].shape}")
    print(f"  VG label shape: {batch['vg_label'].shape}")
    print(f"  Operators keys: {list(batch['operators'].keys())}")
    
    # Test GCNN data loading
    print("\nTesting GCNN (graph) data loading...")
    dm_gcnn = OPFDataModule(
        data_dir=data_dir,
        batch_size=2,
        feature_type="graph",
        feature_params={"feature_iterations": 5},
        normalize=True,
    )
    dm_gcnn.setup(stage="fit")
    
    train_loader = dm_gcnn.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  e_0_k shape: {batch['e_0_k'].shape}")
    print(f"  f_0_k shape: {batch['f_0_k'].shape}")
    print(f"  PD shape: {batch['pd'].shape}")
    print(f"  QD shape: {batch['qd'].shape}")
    
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Complete Training Pipeline Verification")
    print("=" * 60)
    
    success = True
    
    try:
        # 1. Verify data loading
        verify_data_loading()
        
        # 2. Verify DNN pipeline
        dnn_success = verify_dnn_pipeline()
        success = success and dnn_success
        
        # 3. Verify GCNN pipeline
        gcnn_success = verify_gcnn_pipeline()
        success = success and gcnn_success
        
        # Final summary
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        
        if success:
            print("✅ ALL TESTS PASSED!")
            print("\nThe complete pipeline is working correctly:")
            print("  ✓ Data loading with OPFDataModule")
            print("  ✓ Model initialization (DNN and GCNN)")
            print("  ✓ OPFTask with physics loss")
            print("  ✓ PyTorch Lightning training loop")
            print("  ✓ Automatic input format detection")
            print("  ✓ Combined supervised + physics loss")
            sys.exit(0)
        else:
            print("❌ SOME TESTS FAILED")
            print("\nCheck the error messages above for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
