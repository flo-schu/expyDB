from typing import List
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Date, DateTime
from sqlalchemy.orm import Mapped, declarative_base, relationship, mapped_column

# Define your data model
Base = declarative_base()

class Experiment(Base):
    __tablename__ = "experiment_table"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime)
    id_biotox = Column(Integer, nullable=True)
    name = Column(String, nullable=True)
    date = Column(Date, nullable=True)
    experimentator = Column(String, nullable=True)
    treatments: Mapped[List["Treatment"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")
    observations: Mapped[List["Observation"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")

    def __str__(self):
        return f"Experiment(id={self.id}, date={self.date}, experimentator={self.experimentator})"

class Treatment(Base):
    __tablename__ = "treatment_table"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime)
    name = Column(String, nullable=True)
    hpf = Column(Float)
    cext_nom_diuron = Column(Float, default=0.0)
    cext_nom_diclofenac = Column(Float, default=0.0)
    cext_nom_naproxen = Column(Float, default=0.0)
    nzfe = Column(Integer, default=None, nullable=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment_table.id"))
    experiment: Mapped["Experiment"] = relationship(back_populates="treatments")
    observations: Mapped[List["Observation"]] = relationship(back_populates="treatment", cascade="all, delete-orphan")
        
    def __str__(self):
        return f"Treatment(id={self.id}, name={self.name})"

class Observation(Base):
    __tablename__ = "observation_table"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime)
    measurement = Column(String)
    unit = Column(String)
    time = Column(Float)
    replicate_id = Column(Integer, default=0)
    value = Column(Float)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment_table.id"))
    experiment: Mapped["Experiment"] = relationship(back_populates="observations")
    treatment_id: Mapped[int] = mapped_column(ForeignKey("treatment_table.id"))
    treatment: Mapped["Treatment"] = relationship(back_populates="observations")

    def __str__(self):
        return (
            f"Observation(id={self.id}, time={self.time}, "
            f"measurement={self.measurement}, "
            f"unit={self.unit})"
            f"value={round(self.value,6) if self.value is not None else self.value} "
        )
