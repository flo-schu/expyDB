import warnings
import os
from typing import List
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from experimentDB.database_model import Observation, Treatment, Experiment, Base

def label_duplicates(data, index: List[str], duplicate_column="replicate_id"):
    data[duplicate_column] = 0
    for _, group in data.groupby(index):
        if len(group) == 1:
            continue
        for rep, (rkey, _) in enumerate(group.iterrows()):
            data.loc[rkey, duplicate_column] = rep


def setter(variables, identifiers):
    return {key:value for key, value in zip(variables, identifiers)}

def add_data(
    database: str,
    data: pd.DataFrame,
    experiment_variables: List[str],
    treatment_variables: List[str],
    experiment_setter: callable = setter,
    treatment_setter: callable = setter,
):

    nans = data[treatment_variables].isna().values.sum(axis=0)
    if np.any(nans > 0):
        warnings.warn(
            f"NaNs in treatment variables {np.array(treatment_variables)[nans > 0]} detected. " 
            "Fix in data input, define default, or live with nans in treatment info."
        )

    # Create an engine to connect to your database
    CREATED_AT = datetime.datetime.now()
    engine = create_engine(f"sqlite:///{database}", echo=False)

    with Session(engine) as session:

        # group by experiment
        exp_groups = data.groupby(experiment_variables, dropna=False)
        for experiment_identifiers, experiment_rows in exp_groups:
            experiment = Experiment(
                created_at=CREATED_AT,
                **experiment_setter(experiment_variables, experiment_identifiers)
            )

            # group experiments by treatments
            treat_groups = experiment_rows.groupby(treatment_variables, dropna=False)
            for treatment_identifiers, treatment_rows in treat_groups:
                treatment = Treatment(
                    created_at=CREATED_AT,
                    experiment=experiment,
                    **treatment_setter(treatment_variables, treatment_identifiers)
                )

                # assign duplicate keys for repeated measurements
                if "replicate_id" not in treatment_rows.columns:
                    label_duplicates(treatment_rows, index=["time", "measurement"])
                
                # iterate over observations in treatment
                for _, row in treatment_rows.iterrows():
                    observation = Observation(
                        created_at=CREATED_AT,
                        experiment=experiment,
                        treatment=treatment,
                        measurement=row.measurement,
                        unit=row.unit,
                        replicate_id=row.replicate_id,
                        time=row.time,
                        value=row.value  
                    )

                    session.add(observation)

        session.flush()
        session.commit()


def remove_latest(database):
    engine = create_engine(f"sqlite:///{database}", echo=False)
    
    with Session(engine) as session:
        experiments = pd.read_sql(
            select(Experiment), 
            con=f"sqlite:///{database}"
        )

        created_last = experiments.created_at.unique()[-1]
        stmt = select(Experiment).where(Experiment.created_at == created_last)

        for row in session.execute(stmt):
            session.delete(row.Experiment)

        session.flush()
        session.commit()

def create_database(database):
    # SQLAlchemy does not suppor the addition of columns. This has to be done
    # by hand, but this is also not such a big deal. 
    engine = create_engine(f"sqlite:///{database}", echo=False)
    session = Session(engine)
    if not os.path.exists(database):
        Base.metadata.create_all(engine)

def delete_tables(database):
    engine = create_engine(f"sqlite:///{database}", echo=False)
    session = Session(engine)
    if os.path.exists(database):
        Base.metadata.drop_all(engine)


