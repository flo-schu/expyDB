from typing import List, Optional, LiteralString
from datetime import datetime, timedelta

import pandas as pd
import arviz as az
import xarray as xr
import numpy as np

from sqlmodel import ForeignKey, select, inspect, Field, SQLModel, Relationship
from pydantic import field_validator, ValidationInfo, ValidationError

DATETIME_DEFAULT = datetime.now()

class Experiment(SQLModel, table=True):
    id_laboratory: Optional[int] = Field(default=None)
    name: Optional[str] = Field(default=None)
    date: Optional[datetime] = Field(default=datetime(1900,1,1,0,0))
    experimentator: Optional[str] = Field(default=None)
    public: Optional[bool] = Field(default=False)
    info: Optional[str] = Field(default=None, repr=False, description="Extra information about the Experiment")
    
    # meta
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: Optional[datetime] = Field(default=DATETIME_DEFAULT, sa_column_kwargs=dict())
    
    # relationships
    treatments: List["Treatment"] = Relationship(back_populates="experiment", cascade_delete=True)


class Treatment(SQLModel, table=True):
    """The treatment table contains the main pieces of information. In principle,
    all relevant information for repitition of an experiment should be included
    here.

    Any time-variable information that is relevant to the treatment can and should
    be included via the exposures map.
    """
    id: Optional[int] = Field(default=None, primary_key=True, sa_column_kwargs=dict())
    name: Optional[str] = Field(default=None, description="Name of the treatment")
    
    # information about the test subject
    subject: Optional[str] = Field(default=None, description="Identification of the subject of the treatment (Species, Family, Name, ...)")
    subject_age_from: Optional[timedelta] = Field(default=None, description="Age of the test subject, at the start of the treatment")
    subject_age_to: Optional[timedelta] = Field(default=None, description="Age of the test subject, at the start of the treatment")
    subject_count: Optional[float] = Field(default=1, description="Count of the test subjects, if they cannot be discriminated in the experiment")

    # information about the test environment
    medium: Optional[str] = Field(default=None, description="The medium inside the subject lived throughout the treatment")
    volume: Optional[float] = Field(default=None, description="The volume of the medium if applicable.")
    info: Optional[str] = Field(default=None, repr=False, description="Extra information about the treatment")

    # timeseries. This is currently grouped by interventions and observations, however
    interventions: List["Timeseries"] = Relationship(back_populates="treatment", cascade_delete=True, sa_relationship_kwargs=dict())
    observations: List["Timeseries"] = Relationship(back_populates="treatment", cascade_delete=True, sa_relationship_kwargs=dict(overlaps="interventions"))
    
    # relationships to parent tables
    experiment_id: Optional[int] = Field(default=None, foreign_key="experiment.id", sa_column_kwargs=dict())
    experiment: Optional["Experiment"] = Relationship(back_populates="treatments", sa_relationship_kwargs=dict())

    @field_validator("subject_age_from", "subject_age_to", mode="before")
    @classmethod
    def validate_age(cls, value):
        if value is None:
            return value
        else:
            return pd.Timedelta(value)

class Timeseries(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, sa_column_kwargs=dict())
    type: str = Field(description="Can be 'intervention' or 'observation'.")
    variable: str
    name: Optional[str] = Field(default=None, description="e.g. replicate ID")

    # this column is very important, because it requires some thought.
    unit: str
    method: Optional[str] = Field(default=None, description="If type: 'observation': the measurement method. If type: 'intervention': the application method")
    sample: Optional[str] = Field(default=None, description="If type: 'observation', the sample which has been measured. If type: 'intervention', the medium of applying the intervention")
    interpolation: Optional[str] = Field(default="constant", description="How the data are interpolated between timepoints.")
    info: Optional[str] = Field(default=None)

    tsdata: List["TsData"] = Relationship(back_populates="timeseries", sa_relationship_kwargs=dict())

    # relationships to parent tables
    treatment_id: Optional[int] = Field(default=None, foreign_key="treatment.id")
    treatment: Optional["Treatment"] = Relationship(sa_relationship_kwargs=dict())


    @field_validator("method", "sample", "unit", "interpolation", mode="before")
    @classmethod
    def parse_string_to_dict(cls, value, info: ValidationInfo):
        if value is None:
            return None
        # Step 1: Split the input string by commas to get individual components
        pairs = value.split(',')

        # Step 2: Initialize an empty dictionary
        result_dict = {}

        # Step 3: Loop through each part and handle them
        for pair in pairs:
            pair = pair.strip()  # Clean up leading/trailing spaces
            if ':' in pair:
                key, val = pair.split(':', 1)  # Split by the first colon
                key = key.strip()
                val = val.strip()
                result_dict[key] = val
            else:
                # If it's just "topical", treat it as a default entry
                result_dict['default'] = pair.strip()

        # Print the resulting dictionary
        return result_dict.get(info.data["name"], result_dict["default"])

class TsData(SQLModel, table=True):
    """TsData contains only the timestamp and the value associated with the 
    timestamp, all other information:
    - name of the variable (e.g. Food, Diuron, ...)
    - dimension (time, mass, ...)
    - unit (h, mol/L)

    The time information is stored in seconds as this is the most commonly used
    unit in python.

    are assumed constant for any timeseries and stored in the Parent timeseries
    entry.
    """
    id: Optional[int] = Field(primary_key=True, default=None)
    time: timedelta
    value: float

    # relationships to parent tables
    timeseries_id: Optional[int] = Field(default=None, foreign_key="timeseries.id", sa_column_kwargs=dict())
    timeseries: Optional["Timeseries"] = Relationship(back_populates="tsdata", sa_relationship_kwargs=dict())

    @field_validator("time", mode="before")
    @classmethod
    def to_timedelta(cls, value, info):
        if isinstance(value, float|int):
            return timedelta(seconds=float(value))
        elif isinstance(value, timedelta):
            return value
        elif isinstance(value, np.timedelta64):
            return value.item()
        elif isinstance(value, pd.Timedelta):
            return value.to_pytimedelta()
        else:
            raise TypeError(
                "TsData expects 'time' Field to be of types: "
                "'float', 'datetime.timedelta', 'pd.Timedelta' or 'np.timedelta64'"
            )
def split_join(df, statement):
    """Corrects column names from informations contained in the statement"""
    from_clause = statement.froms[0]

    left =  from_clause.left
    right = from_clause.right

    model_a = statement.column_descriptions[0]["type"]
    model_b = statement.column_descriptions[1]["type"]

    name_a = model_a.__name__
    columns_a = [c.name for c in list(inspect(model_a).c)]

    name_b = model_b.__name__
    columns_b = [c.name for c in list(inspect(model_b).c)]

    frame_a = df.iloc[:, 0:len(columns_a)]
    frame_b = df.iloc[:, len(columns_a):]
    frame_b.columns = columns_b
    return (name_a, frame_a), (name_b, frame_b)

def rename_duplicate_columns(columns, compare_table, prefix):
    new_columns = []
    for c in columns:
        if c in compare_table.columns:
            new_columns.append(prefix + c)
        else:
            new_columns.append(c)

    return new_columns


def from_expydb(database, statement=None):
    """
    The data selection method uses the intervention model to load 
    data from an intervention database.

    It uses a SQL statement to select data from the database, which is
    very human readable. 

    It is currently not supported to change the select and join statement,
    but the where statement can be modified at will. Use sqlalchemy methods
    or_, and_, not_, etc. for this.

    This API should satisfy almost any needs. Currently it does not include
    experiment information or selection by entire experiments, which should
    be implemented. For this all the tables Experiment, Treatment and
    Timeseries should be joined.

    If post query selections need to be performed this could be done by
    the select API, but this is currently highly experimental.

    idata objects can be stored and contain all relevant information about
    the treatment and contain all relevant information

    Obtains all information associated with the queried treatments.
    If more than one intervention occurs in the treatment, this will be retrieved
    as an interverntion as well.

    statement -> Treatments -> [observations, interventions]

    if the statement leads to the inclusion of duplicate treatments these are 
    removed, so that timeseries are not duplicated.

    Example
    -------

    """
    if statement is None:
        statement = (
            select(Timeseries, Treatment)
            .join(Timeseries.treatment,)
        )

    # get the joint table and split it according to the database model
    joint_table = pd.read_sql(statement, con=database)
    (_, timeseries_table), (_, treatment_table) = split_join(joint_table, statement)

    # names including table according to join statement
    full_names = statement.froms[0].columns.keys()
    joint_table.columns = full_names

    # all treatments associated with the queried variable
    treatment_table.columns = rename_duplicate_columns(
        columns=treatment_table.columns,
        compare_table=timeseries_table,
        prefix="treatment_"
    )

    # drop duplicate treatments.
    duplicates = treatment_table.duplicated(["treatment_id"], keep="first")
    treatment_table = treatment_table[~duplicates]
    timeseries_table = timeseries_table[~duplicates]
    

    intervention_timeseries = {}
    observation_timseries = {}
    for _, treatment_row in treatment_table.iterrows():
        # query all timeseries of the obtained treatment ids
        # the latter join Table always receives _1 as a postfix when duplicate names
        # occurr
        statement = (
            select(Treatment, Timeseries)
            .join(Timeseries.treatment)
            .where(Treatment.id == treatment_row.treatment_id)
        )

        joint_table = pd.read_sql(statement, con=database)
        (_, treatment_table_), (_, timeseries_table) = split_join(joint_table, statement)

        timeseries_table.columns = rename_duplicate_columns(
            columns=timeseries_table.columns,
            compare_table=treatment_table_,
            prefix="timeseries_"
        )
        timeseries_table = timeseries_table.drop(columns="treatment_id")
        for _, timeseries_row in timeseries_table.iterrows():
            # query all TsDatasets in the obtained timeseries
            statement = (
                select(TsData, Timeseries)
                .join(Timeseries.tsdata)
                .where(Timeseries.id == timeseries_row.timeseries_id)
            )
            joint_table = pd.read_sql(statement, con=database)
            (_, tsdata_table), (_, _) = split_join(joint_table, statement)

            tsdata_table = tsdata_table.drop(columns="timeseries_id")
            variable = timeseries_row.pop("variable")

            ts_data = tsdata_table.set_index("time").to_xarray()
            ts_data = ts_data.assign_coords(id=ts_data["id"])
            ts_data = ts_data.rename(value=variable)
            ts_data = ts_data.expand_dims("timeseries_id")
            ts_data = ts_data.assign_coords({key: ("timeseries_id", list([val])) for key, val in timeseries_row.items()})
            ts_data = ts_data.assign_coords({key: ("timeseries_id", list([val])) for key, val in treatment_row.items()})



            if timeseries_row["type"] == "observation":
                if variable not in observation_timseries:        
                    observation_timseries.update({variable: [ts_data]})
                else:
                    observation_timseries[variable].append(ts_data)
            elif timeseries_row["type"] == "intervention":
                if variable not in intervention_timeseries:        
                    intervention_timeseries.update({variable: [ts_data]})
                else:
                    intervention_timeseries[variable].append(ts_data)
            else:
                raise NotImplementedError(
                    "New data model type has not been implemented in this Case-study."
                )

    observations_ = {variable: xr.concat(arr, dim="timeseries_id")
                    for variable, arr in observation_timseries.items()}

    interventions_ = {variable: xr.concat(arr, dim="timeseries_id")
                    for variable, arr in intervention_timeseries.items()}

    observations = az.InferenceData(**observations_)
    interventions = az.InferenceData(**interventions_)

    return observations, interventions



def add_tsdata(df: pd.DataFrame, time_unit: str, timeseries: Timeseries):
    """This could be used in read_timeseries instead"""
    if time_unit == "d" or time_unit == "day" or time_unit == "days":
        time_unit = "days"
    elif time_unit == "h" or time_unit == "hour" or time_unit == "hours":
        time_unit = "hours"
    elif time_unit == "m" or time_unit == "minute" or time_unit == "minutes":
        time_unit = "minutes"
    elif time_unit == "s" or time_unit == "second" or time_unit == "seconds":
        time_unit = "seconds"
    else:
        raise RuntimeError(
            f"time_unit: {time_unit} is not implemented. Use one of "
            f"days, hours, minutes, or seconds."
        )
    
    if np.issubdtype(df["time"].dtype, np.datetime64): # type: ignore
        df.loc[:,"time"] = df.time - df.time.iloc[0]

    for _, row in df.iterrows():
        if isinstance(row.time, float|int):
            time = timedelta(**{time_unit: row.time})
        else:
            time = row.time

        ts_data = TsData(
            time=time,
            value=row.value,
        )

        timeseries.tsdata.append(ts_data)


def to_expydb(interventions, observations, meta, time_units) -> Experiment:
    """This method takes the metadata and groups them in sections
    - Experiment
    - Treatment
    - Timeseries

    The goal is to name the rows identical to the keys in the Models plus
    some syntactic sugar.
    Exposure path -> exposure_path

    Then I can group the sections and simply pass them as keyword arguments
    to the models. If they are not set, fine. Then the error will be raised
    at this stage; and I should only be required to complete the necessary metadata

    The file should contain a sheet for each intervention and a sheet for each 
    observation that should be tracked
    """
    CREATED_AT = datetime.now()
    meta.index = meta.index.str.lower().str.replace(" ","_")
    meta_ = meta.copy().loc[:,"Value"].T
    meta_full = meta.copy().loc[:,"Notes"].T

    default_time_unit = meta_.get("time_unit", default="days")

    # get experiment fields
    experiment_columns = [
        column for column in Experiment.__fields__
        if not (
            column.primary_key or 
            column.key in ["created_at", "info"]
        )
    ]

    experiment_fields = {k: meta_.get(k) for k in experiment_columns}

    # get treatment fields
    treatment_columns = [
        column.key for column in Treatment.__table__.columns
        if not (
            column.primary_key or 
            column.key in ["name", "created_at", "experiment_id", "info"]
        )
    ]
    treatment_fields = {k: meta_.get(k) for k in treatment_columns}

    # get intervention timeseries fields
    intervention_timeseries_columns = [
        column.key for column in Timeseries.__table__.columns
        if not (
            column.primary_key or 
            column.key in ["name", "type", "variable", "created_at", "treatment_id", "experiment_id", "info"]
        )
    ]
    intervention_timeseries_fields = {k: meta_.get(f"intervention_{k}") for k in intervention_timeseries_columns}

    # get observation timeseries fields
    observation_timeseries_columns = [
        column.key for column in Timeseries.__table__.columns
        if not (
            column.primary_key or 
            column.key in ["name", "type", "variable", "created_at", "treatment_id", "experiment_id", "info"]
        )
    ]
    observation_timeseries_fields = {k: meta_.get(f"observation_{k}") for k in observation_timeseries_columns}


    # remaining meta to a readable string
    info = meta_full.to_json()#.replace("\n", "---")


    experiment = Experiment(**experiment_fields, info=info)
            
    experiment.created_at=CREATED_AT
    
    for (tid, observation_group), (tid, intervention_group) in zip(
        observations.groupby("treatment_id"),
        interventions.groupby("treatment_id")
    ):
        # add treatment
        treatment = Treatment(
            name=str(tid),
            **treatment_fields,
        )

        experiment.treatments.append(treatment)

        # TODO: Test does not pass because of multi exposure
        # test_equality_of_exposure_patterns_in_treatment(df=exposure_group)
        # intervention_pattern = list(intervention_group.groupby("replicate_id"))[0][1]

        # add exposure interventions
        interventions = [s.strip(" ") for s in meta_["interventions"].split(",")]
        for iv in interventions:

            ts_exposure = Timeseries(
                type="intervention",
                variable=iv,
                **intervention_timeseries_fields
            )

            treatment.interventions.append(ts_exposure)
            time_unit = time_units["interventions"][iv]
            tsdata_iv = intervention_group[["time", "treatment_id", "replicate_id", iv]]
            tsdata_iv = tsdata_iv.rename(columns={iv: "value"})
            add_tsdata(
                df=tsdata_iv, 
                time_unit=time_unit if bool(time_unit) else default_time_unit, 
                timeseries=ts_exposure
            )

        obsevations_pattern = list(observation_group.groupby("replicate_id"))[0][1]

        # add observations
        observations = [s.strip(" ") for s in meta_["observations"].split(",")]

        for rep_id, observation_rep in observation_group.groupby("replicate_id"):
            for obs in observations:
                ts_survival_rep = Timeseries(
                    name=str(rep_id),
                    type="observation",
                    variable=obs,
                    **observation_timeseries_fields
                )

                treatment.observations.append(ts_survival_rep)

                time_unit = time_units["observations"][obs]
                tsdata_obs = observation_rep[["time", "treatment_id", "replicate_id", obs]]
                tsdata_obs = tsdata_obs.rename(columns={obs: "value"})
                add_tsdata(
                    df=tsdata_obs, 
                    time_unit=time_unit, 
                    timeseries=ts_survival_rep
                )

    return experiment
