from typing import List, Optional
from datetime import datetime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import (
    relationship, 
    mapped_column, 
    Mapped, 
    MappedAsDataclass,
    DeclarativeBase,
)

# declarative base class
class Base(MappedAsDataclass, DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiment_table"
    
    id_laboratory: Mapped[Optional[int]] = mapped_column(default=None)
    name: Mapped[Optional[str]] = mapped_column(default=None)
    date: Mapped[Optional[datetime]] = mapped_column(default=datetime(1900,1,1,0,0))
    experimentator: Mapped[str] = mapped_column(default=None)
    
    # meta
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(init=False)
    
    # relationships
    treatments: Mapped[List["Treatment"]] = relationship(init=False, repr=False, back_populates="experiment", cascade="all, delete-orphan")
    observations: Mapped[List["Observation"]] = relationship(init=False, repr=False, back_populates="experiment", cascade="all, delete-orphan")


class Treatment(Base):
    __tablename__ = "treatment_table"
    
    hpf: Mapped[float]
    name: Mapped[Optional[str]] = mapped_column(default=None)
    cext_nom_diuron: Mapped[Optional[float]] = mapped_column(default=0.0)
    cext_nom_diclofenac: Mapped[Optional[float]] = mapped_column(default=0.0)
    cext_nom_naproxen: Mapped[Optional[float]] = mapped_column(default=0.0)
    nzfe: Mapped[Optional[int]] = mapped_column(default=None)
    
    # meta
    id: Mapped[int] = mapped_column(init=False, primary_key=True)

    # relationships    
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment_table.id"), init=False)
    experiment: Mapped["Experiment"] = relationship(init=False, repr=False, back_populates="treatments")
    observations: Mapped[List["Observation"]] = relationship(init=False, repr=False, back_populates="treatment", cascade="all, delete-orphan")


class Observation(Base):
    __tablename__ = "observation_table"

    measurement: Mapped[str]
    unit: Mapped[str]
    time: Mapped[float]
    method: Mapped[Optional[str]] = mapped_column(default=None)
    value: Mapped[Optional[float]] = mapped_column(default=None)
    replicate_id: Mapped[Optional[int]] = mapped_column(default=0)

    # meta
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    
    # relationships
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment_table.id"), init=False)
    treatment_id: Mapped[int] = mapped_column(ForeignKey("treatment_table.id"), init=False)
    experiment: Mapped["Experiment"] = relationship(back_populates="observations", repr=False, init=False)
    treatment: Mapped["Treatment"] = relationship(back_populates="observations", repr=False, init=False)
    