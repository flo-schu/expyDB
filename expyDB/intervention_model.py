from typing import List, Optional, Timedelta
from datetime import datetime, timedelta
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
    public: Mapped[bool] = mapped_column(default=False)
    
    # meta
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(init=False)
    
    # relationships
    treatments: Mapped[List["Treatment"]] = relationship(init=False, repr=False, back_populates="experiment", cascade="all, delete-orphan")


class Treatment(Base):
    """The treatment table contains the main pieces of information. In principle,
    all relevant information for repitition of an experiment should be included
    here.

    Any time-variable information that is relevant to the treatment can and should
    be included via the exposures map.
    """
    __tablename__ = "treatment_table"
    
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(default=None, doc="Name of the treatment")
    
    # information about the test subject
    subject: Mapped[str]
    subject_age: Mapped[Optional[timedelta]] = mapped_column(default=None, doc="Age of the test subject, at the start of the treatment")
    subject_count: Mapped[Optional[int]] = mapped_column(default=1, doc="Count of the test subjects, if they cannot be discriminated in the experiment")

    # information about the test environment
    medium: Mapped[Optional[str]] = mapped_column(default=None, doc="The medium inside the subject lived throughout the treatment")
    volume: Mapped[Optional[float]] = mapped_column(default=None, doc="The volume of the medium if applicable.")
    info: Mapped[Optional[str]] = mapped_column(default=None, doc="Extra information about the treatment")

    # timeseries. This is currently grouped by exposures and observations, however
    # these could technically be grouped into on relational mapping (timeseries)
    # TODO: Find out which is the better way to handle such data.
    interventions: Mapped[List["Timeseries"]] = relationship(init=False, repr=False, back_populates="treatment", cascade="all, delete-orphan")
    observations: Mapped[List["Timeseries"]] = relationship(init=False, repr=False, back_populates="treatment", cascade="all, delete-orphan")
    
    # relationships to parent tables
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment_table.id"), init=False)
    experiment: Mapped["Experiment"] = relationship(init=False, repr=False, back_populates="treatments")


class Timeseries(Base):
    __tablename__ = "timeseries_table"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    type: Mapped[str] = mapped_column(doc="Can be 'intervention' or 'observation'.")
    variable: Mapped[str]
    dimension: Mapped[str]
    unit: Mapped[str]
    sample: Mapped[Optional[str]] = mapped_column(default=None, doc="If type 'observation', the sample which has been measured.")
    location: Mapped[Optional[str]] = mapped_column(default=None, doc="Where the observation has been made")
    method: Mapped[Optional[str]] = mapped_column(default=None, doc="The measurement method")
    interpolation: Mapped[str] = mapped_column(default="constant", doc="How the data are interpolated between timepoints.")
    info: Mapped[str] = mapped_column(default=None)

    tsdata: Mapped[List["TsData"]] = relationship(back_populates="timeseries", repr=False, init=False)

    # relationships to parent tables
    treatment_id: Mapped[Optional[int]] = mapped_column(ForeignKey("treatment_table.id"), init=False)
    # TODO: Not sure how back-population works with two columns that refer to the same table
    treatment: Mapped[Optional["Treatment"]] = relationship(repr=False, init=False)


class TsData(Base):
    """TsData contains only the timestamp and the value associated with the 
    timestamp, all other information:
    - name of the variable (e.g. Food, Diuron, ...)
    - dimension (time, mass, ...)
    - unit (h, mol/L)
    are assumed constant for any timeseries and stored in the Parent timeseries
    entry.
    """
    __tablename__ = "tsdata_table"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    time: Mapped[timedelta] = mapped_column(default=0)
    value: Mapped[float]

    # relationships to parent tables
    timeseries_id = Mapped[int] = mapped_column(ForeignKey("timeseries_table.id"), init=False)
    timeseries: Mapped["Timeseries"] = relationship(back_populates="tsdata", repr=False, init=False)
