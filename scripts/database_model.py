from typing import List
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Date
from sqlalchemy.orm import Mapped, declarative_base, relationship, mapped_column

# Define your data model
Base = declarative_base()

class Experiment(Base):
    __tablename__ = "experiment_table"
    id = Column(Integer, primary_key=True)
    id_biotox = Column(Integer, nullable=True)
    name = Column(String, nullable=True)
    date = Column(Date, nullable=True)
    experimentator = Column(String, nullable=True)
    treatments: Mapped[List["Treatment"]] = relationship(back_populates="experiment")
    observations: Mapped[List["Observation"]] = relationship(back_populates="experiment")

class Treatment(Base):
    __tablename__ = "treatment_table"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    hpf = Column(Float)
    cext_nom_diuron = Column(Float, default=0.0)
    cext_nom_diclofenac = Column(Float, default=0.0)
    cext_nom_naproxen = Column(Float, default=0.0)
    nzfe = Column(Integer, default=None, nullable=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment_table.id"))
    experiment: Mapped["Experiment"] = relationship(back_populates="treatments")
    observations: Mapped[List["Observation"]] = relationship(back_populates="treatment")

class Observation(Base):
    __tablename__ = "observation_table"
    id = Column(Integer, primary_key=True)
    measurement = Column(String)
    unit = Column(String)
    time = Column(Float)
    replicate_id = Column(Integer, default=0)
    value = Column(Float)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment_table.id"))
    experiment: Mapped["Experiment"] = relationship(back_populates="observations")
    treatment_id: Mapped[int] = mapped_column(ForeignKey("treatment_table.id"))
    treatment: Mapped["Treatment"] = relationship(back_populates="observations")

