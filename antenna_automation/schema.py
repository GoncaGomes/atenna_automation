"""Pydantic models describing the structured antenna extraction output."""

from __future__ import annotations

from typing import Literal, Optional, List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictBaseModel(BaseModel):
    """Base model that forbids undeclared fields to keep the schema tight."""

    model_config = ConfigDict(extra="forbid")


class Point3D(StrictBaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class Provenance(StrictBaseModel):
    source_file: str = Field(description="e.g., 'pages/page_0006.txt' or 'figures/fig_003.png'")
    page_or_fig: str = Field(description="e.g., 'page 6', 'fig 3'")
    snippet: Optional[str] = Field(default=None, description="Short quote used as evidence")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class MaterialProps(StrictBaseModel):
    name: Optional[str] = Field(default=None, description="e.g., 'Rogers RO3010'")
    epsilon_r: Optional[float] = Field(default=None, description="Relative permittivity εr")
    tan_delta: Optional[float] = Field(default=None, description="Loss tangent")
    conductivity_S_per_m: Optional[float] = Field(default=None, description="Metal conductivity if applicable")


class Layer(StrictBaseModel):
    order: int = Field(description="0 = bottom-most in the stackup (in +z order)")
    role: str = Field(description="Functional role of the layer")
    material: MaterialProps = Field(description="Material and its RF properties if available")
    z0: float = Field(description="Height from where the layer begins (z-axis) in mm. If this is the bottom layer, z0=0.")
    thickness_mm: Optional[float] = Field(default=None, description="Physical thickness in mm")
    pattern: Optional[str] = Field(default=None, description="e.g., 'solid', 'etched', 'slotted'")
    geometry: Optional[str] = Field(
        default=None,
        description=(
            "Describe the geometry of this layer. All relative to the one before. "
            "Include dimensions and shapes when stated."
        ),
    )
    provenance: Optional[Provenance] = None


class Polarization(StrictBaseModel):
    type: str = Field(description="Polarization type: e.g., 'linear', 'circular', 'elliptical'")
    sense: Optional[Literal["RHCP", "LHCP"]] = Field(
        default=None, description="When circular, the handedness"
    )


class OperatingPoint(StrictBaseModel):
    freq_GHz: float = Field(description="Operating or reported frequency in GHz")
    gain_dBi: Optional[float] = Field(default=None, description="Reported gain in dBi at the operating frequency")
    axial_ratio_dB: Optional[float] = Field(default=None, description="Reported axial ratio in dB at the operating frequency")
    bandwidth_MHz: Optional[float] = Field(default=None, description="3-dB AR or |S11| bandwidth if stated")
    provenance: Optional[Provenance] = None


class Feed(StrictBaseModel):
    type: str = Field(description="Feeding approach, e.g., 'coax', 'microstrip line', 'SMA probe'")
    dimensions_mm: Optional[str] = Field(
        default=None,
        description="Physical dimensions (length, diameter) in mm if specified",
    )
    location_mm: Point3D = Field(
        default=None,
        description="Coordinates relative to patch/board origin in mm when available",
    )
    provenance: Optional[Provenance] = None


class Antenna(StrictBaseModel):
    antenna_type: str = Field(description="The type of antenna e.g., 'microstrip patch', 'dipole', 'array'")
    polarization: Polarization = Field(description="Polarization type (and sense if circular)")
    operating_points: List[OperatingPoint] = Field(description="Per-frequency metrics")
    stackup: List[Layer] = Field(description="Layered description from bottom (order=0) to top")
    feed: Optional[Feed] = None
    notes: Optional[str] = None

    @field_validator("operating_points")
    @classmethod
    def _non_empty_ops(cls, value: List[OperatingPoint]) -> List[OperatingPoint]:
        if not value:
            raise ValueError("Provide at least one OperatingPoint, even if freq only.")
        return value

