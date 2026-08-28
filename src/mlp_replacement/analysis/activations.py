"""Activation geometry and reconstruction utilities for block analysis."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActivationSpectrum:
    """Summarize the centered covariance spectrum of token activations."""

    num_tokens: int
    dimension: int
    eigenvalues: torch.Tensor
    effective_rank: float
    stable_rank: float
    k90: int
    k95: int
    k99: int
    mean_energy_fraction: float

    def to_dict(self, include_eigenvalues=True):
        """Convert the summary to JSON-compatible values."""

        result = {
            "num_tokens": self.num_tokens,
            "dimension": self.dimension,
            "effective_rank": self.effective_rank,
            "normalized_effective_rank": self.effective_rank / self.dimension,
            "stable_rank": self.stable_rank,
            "normalized_stable_rank": self.stable_rank / self.dimension,
            "k90": self.k90,
            "k95": self.k95,
            "k99": self.k99,
            "mean_energy_fraction": self.mean_energy_fraction,
        }
        if include_eigenvalues:
            result["eigenvalues"] = self.eigenvalues.tolist()
        return result


@dataclass(frozen=True)
class CovarianceEigendecomposition:
    """Store a calibration mean and descending covariance eigenpairs."""

    mean: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor | None


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Record reconstruction errors at one retained dimension."""

    method: str
    rank: int
    activation_relative_mse: float
    output_relative_mse: float | None


def _validate_activations(activations):
    if activations.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional token-by-feature matrix, got {activations.shape}"
        )
    if activations.shape[0] < 2:
        raise ValueError("At least two activation vectors are required")
    if activations.shape[1] < 1:
        raise ValueError("Activation dimension must be positive")
    if not torch.isfinite(activations).all():
        raise ValueError("Activation matrix contains non-finite values")


def covariance_eigendecomposition(activations, device=None, with_eigenvectors=False):
    """Compute a float32 centered covariance eigendecomposition.

    The input may remain on CPU while the calculation is performed on a selected
    accelerator. Results are returned on CPU so they do not retain GPU memory.
    """

    _validate_activations(activations)
    calculation_device = torch.device(device) if device is not None else activations.device
    values = activations.to(device=calculation_device, dtype=torch.float32)
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / (values.shape[0] - 1)
    if with_eigenvectors:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.arange(
            eigenvalues.numel() - 1,
            -1,
            -1,
            device=eigenvalues.device,
        )
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        eigenvectors = eigenvectors[:, order]
        vectors_cpu = eigenvectors.cpu()
    else:
        eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0.0)
        vectors_cpu = None
    return CovarianceEigendecomposition(
        mean=mean.cpu(),
        eigenvalues=eigenvalues.cpu(),
        eigenvectors=vectors_cpu,
    )


def explained_variance_dimension(eigenvalues, threshold):
    """Return the smallest component count meeting a variance threshold."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("Explained-variance threshold must be in (0, 1]")
    total = eigenvalues.sum()
    if total <= 0:
        return 0
    cumulative = eigenvalues.cumsum(0) / total
    index = torch.searchsorted(
        cumulative,
        torch.tensor(
            threshold,
            dtype=cumulative.dtype,
            device=cumulative.device,
        ),
    )
    return min(int(index.item()) + 1, eigenvalues.numel())


def summarize_activation_spectrum(activations, decomposition=None, device=None):
    """Compute participation-ratio and stable-rank activation diagnostics."""

    _validate_activations(activations)
    if decomposition is None:
        decomposition = covariance_eigendecomposition(
            activations,
            device=device,
            with_eigenvectors=False,
        )
    eigenvalues = decomposition.eigenvalues.float().clamp_min(0.0)
    total = float(eigenvalues.sum().item())
    squared_total = float(eigenvalues.square().sum().item())
    largest = float(eigenvalues[0].item()) if eigenvalues.numel() else 0.0
    effective_rank = total * total / squared_total if squared_total > 0 else 0.0
    stable_rank = total / largest if largest > 0 else 0.0

    values = activations.float()
    mean_energy = float(values.mean(dim=0).square().sum().item())
    total_second_moment = float(values.square().sum(dim=1).mean().item())
    mean_energy_fraction = (
        mean_energy / total_second_moment if total_second_moment > 0 else 0.0
    )
    return ActivationSpectrum(
        num_tokens=int(activations.shape[0]),
        dimension=int(activations.shape[1]),
        eigenvalues=eigenvalues,
        effective_rank=effective_rank,
        stable_rank=stable_rank,
        k90=explained_variance_dimension(eigenvalues, 0.90),
        k95=explained_variance_dimension(eigenvalues, 0.95),
        k99=explained_variance_dimension(eigenvalues, 0.99),
        mean_energy_fraction=mean_energy_fraction,
    )


def coordinate_orders(calibration, seed=21):
    """Return nested random and top-variance coordinate orderings."""

    _validate_activations(calibration)
    variances = calibration.float().var(dim=0, unbiased=True)
    top_variance = torch.argsort(variances, descending=True).cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_order = torch.randperm(calibration.shape[1], generator=generator)
    return {
        "random_coordinates": random_order,
        "top_variance_coordinates": top_variance,
    }


def _relative_error(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else 0.0


def evaluate_reconstruction(
    values,
    mean,
    ranks,
    *,
    method,
    basis=None,
    coordinate_order=None,
    output_weight=None,
    output_bias=None,
    device=None,
    batch_size=512,
):
    """Evaluate PCA or coordinate-retention reconstruction on held-out values.

    Discarded coordinates are reconstructed with the calibration mean, making
    coordinate baselines comparable to centered PCA reconstruction.
    """

    _validate_activations(values)
    dimension = int(values.shape[1])
    requested_ranks = tuple(sorted(set(int(rank) for rank in ranks)))
    if not requested_ranks or requested_ranks[0] < 1 or requested_ranks[-1] > dimension:
        raise ValueError(f"Ranks must lie within [1, {dimension}]")
    if (basis is None) == (coordinate_order is None):
        raise ValueError("Provide exactly one of basis or coordinate_order")
    if mean.ndim != 1 or mean.numel() != dimension:
        raise ValueError("Reconstruction mean does not match activation dimension")
    if basis is not None and (
        basis.ndim != 2
        or basis.shape[0] != dimension
        or basis.shape[1] < requested_ranks[-1]
    ):
        raise ValueError("PCA basis dimension does not match activation dimension")
    if coordinate_order is not None and coordinate_order.numel() != dimension:
        raise ValueError("Coordinate order must contain every activation coordinate")
    if coordinate_order is not None:
        sorted_coordinates = coordinate_order.detach().cpu().long().sort().values
        if not torch.equal(sorted_coordinates, torch.arange(dimension)):
            raise ValueError("Coordinate order must be a permutation of all coordinates")
    if output_weight is not None and (
        output_weight.ndim != 2 or output_weight.shape[1] != dimension
    ):
        raise ValueError("Output projection input dimension does not match activations")

    calculation_device = torch.device(device) if device is not None else values.device
    mean_device = mean.to(device=calculation_device, dtype=torch.float32)
    weight_device = (
        output_weight.to(device=calculation_device, dtype=torch.float32)
        if output_weight is not None
        else None
    )
    bias_device = (
        output_bias.to(device=calculation_device, dtype=torch.float32)
        if output_bias is not None
        else None
    )
    results = []

    for rank in requested_ranks:
        if basis is not None:
            retained_basis = basis[:, :rank].to(
                device=calculation_device,
                dtype=torch.float32,
            )
            retained_indices = None
        else:
            retained_basis = None
            retained_indices = coordinate_order[:rank].to(calculation_device)

        activation_error = 0.0
        activation_energy = 0.0
        output_error = 0.0
        output_energy = 0.0
        for start in range(0, values.shape[0], batch_size):
            batch = values[start : start + batch_size].to(
                device=calculation_device,
                dtype=torch.float32,
            )
            centered = batch - mean_device
            if retained_basis is not None:
                reconstruction = (
                    centered @ retained_basis @ retained_basis.T + mean_device
                )
            else:
                reconstruction = mean_device.expand_as(batch).clone()
                reconstruction[:, retained_indices] = batch[:, retained_indices]

            activation_error += float((reconstruction - batch).square().sum().item())
            activation_energy += float(batch.square().sum().item())
            if weight_device is not None:
                target_output = batch @ weight_device.T
                reconstructed_output = reconstruction @ weight_device.T
                if bias_device is not None:
                    target_output = target_output + bias_device
                    reconstructed_output = reconstructed_output + bias_device
                output_error += float(
                    (reconstructed_output - target_output).square().sum().item()
                )
                output_energy += float(target_output.square().sum().item())

        results.append(
            ReconstructionMetrics(
                method=method,
                rank=rank,
                activation_relative_mse=_relative_error(
                    activation_error,
                    activation_energy,
                ),
                output_relative_mse=(
                    _relative_error(output_error, output_energy)
                    if weight_device is not None
                    else None
                ),
            )
        )
    return tuple(results)
